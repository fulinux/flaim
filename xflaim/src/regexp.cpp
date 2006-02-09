//------------------------------------------------------------------------------
// Desc:	This is regular expression class.
//
// Tabs:	3
//
//		Copyright (c) 2004-2006 Novell, Inc. All Rights Reserved.
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
// $Id: regexp.cpp 3116 2006-01-19 13:31:53 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

typedef enum
{
	EXP_LITERAL = 0,
	EXP_CHAR_CLASS,
	EXP_ALTERNATIVES
} eExpType;

typedef struct BlockCharRangeTag
{
	FLMUNICODE		uzLowChar;
	FLMUNICODE		uzHighChar;
	const char *	pszBlockName;
} BLOCK_CHAR_RANGE;

BLOCK_CHAR_RANGE FlmBlockCharRanges[] =
{
	{0x0000, 0x007F, "BasicLatin"},
	{0x0080, 0x00FF, "Latin-1Supplement"},
	{0x0100, 0x017F, "LatinExtended-A"},
	{0x0180, 0x024F, "LatinExtended-B"},
	{0x0250, 0x02AF, "IPAExtensions"},
	{0x02B0, 0x02FF, "SpacingModifierLetters"},
	{0x0300, 0x036F, "CombiningDiacriticalMarks"},
	{0x0370, 0x03FF, "Greek"},
	{0x0400, 0x04FF, "Cyrillic"},
	{0x0530, 0x058F, "Armenian"},
	{0x0590, 0x05FF, "Hebrew"},
	{0x0600, 0x06FF, "Arabic"},
	{0x0700, 0x074F, "Syriac"},
	{0x0780, 0x07BF, "Thaana"},
	{0x0900, 0x097F, "Devanagari"},
	{0x0980, 0x09FF, "Bengali"},
	{0x0A00, 0x0A7F, "Gurmukhi"},
	{0x0A80, 0x0AFF, "Gujarati"},
	{0x0B00, 0x0B7F, "Oriya"},
	{0x0B80, 0x0BFF, "Tamil"},
	{0x0C00, 0x0C7F, "Telugu"},
	{0x0C80, 0x0CFF, "Kannada"},
	{0x0D00, 0x0D7F, "Malayalam"},
	{0x0D80, 0x0DFF, "Sinhala"},
	{0x0E00, 0x0E7F, "Thai"},
	{0x0E80, 0x0EFF, "Lao"},
	{0x0F00, 0x0FFF, "Tibetan"},
	{0x1000, 0x109F, "Myanmar"},
	{0x10A0, 0x10FF, "Georgian"},
	{0x1100, 0x11FF, "HangulJamo"},
	{0x1200, 0x137F, "Ethiopic"},
	{0x13A0, 0x13FF, "Cherokee"},
	{0x1400, 0x167F, "UnifiedCanadianAboriginalSyllabics"},
	{0x1680, 0x169F, "Ogham"},
	{0x16A0, 0x16FF, "Runic"},
	{0x1780, 0x17FF, "Khmer"},
	{0x1800, 0x18AF, "Mongolian"},
	{0x1E00, 0x1EFF, "LatinExtendedAdditional"},
	{0x1F00, 0x1FFF, "GreekExtended"},
	{0x2000, 0x206F, "GeneralPunctuation"},
	{0x2070, 0x209F, "SuperscriptsandSubscripts"},
	{0x20A0, 0x20CF, "CurrencySymbols"},
	{0x20D0, 0x20FF, "CombiningMarksforSymbols"},
	{0x2100, 0x214F, "LetterlikeSymbols"},
	{0x2150, 0x218F, "NumberForms"},
	{0x2190, 0x21FF, "Arrows"},
	{0x2200, 0x22FF, "MathematicalOperators"},
	{0x2300, 0x23FF, "MiscellaneousTechnical"},
	{0x2400, 0x243F, "ControlPictures"},
	{0x2440, 0x245F, "OpticalCharacterRecognition"},
	{0x2460, 0x24FF, "EnclosedAlphanumerics"},
	{0x2500, 0x257F, "BoxDrawing"},
	{0x2580, 0x259F, "BlockElements"},
	{0x25A0, 0x25FF, "GeometricShapes"},
	{0x2600, 0x26FF, "MiscellaneousSymbols"},
	{0x2700, 0x27BF, "Dingbats"},
	{0x2800, 0x28FF, "BraillePatterns"},
	{0x2E80, 0x2EFF, "CJKRadicalsSupplement"},
	{0x2F00, 0x2FDF, "KangxiRadicals"},
	{0x2FF0, 0x2FFF, "IdeographicDescriptionCharacters"},
	{0x3000, 0x303F, "CJKSymbolsandPunctuation"},
	{0x3040, 0x309F, "Hiragana"},
	{0x30A0, 0x30FF, "Katakana"},
	{0x3100, 0x312F, "Bopomofo"},
	{0x3130, 0x318F, "HangulCompatibilityJamo"},
	{0x3190, 0x319F, "Kanbun"},
	{0x31A0, 0x31BF, "BopomofoExtended"},
	{0x3200, 0x32FF, "EnclosedCJKLettersandMonths"},
	{0x3300, 0x33FF, "CJKCompatibility"},
	{0x3400, 0x4DB5, "CJKUnifiedIdeographsExtensionA"},
	{0x4E00, 0x9FFF, "CJKUnifiedIdeographs"},
	{0xA000, 0xA48F, "YiSyllables"},
	{0xA490, 0xA4CF, "YiRadicals"},
	{0xAC00, 0xD7A3, "HangulSyllables"},
	{0xD800, 0xDB7F, "HighSurrogates"},
	{0xDB80, 0xDBFF, "HighPrivateUseSurrogates"},
	{0xDC00, 0xDFFF, "LowSurrogates"},
	{0xE000, 0xF8FF, "PrivateUse"},
	{0xF900, 0xFAFF, "CJKCompatibilityIdeographs"},
	{0xFB00, 0xFB4F, "AlphabeticPresentationForms"},
	{0xFB50, 0xFDFF, "ArabicPresentationForms-A"},
	{0xFE20, 0xFE2F, "CombiningHalfMarks"},
	{0xFE30, 0xFE4F, "CJKCompatibilityForms"},
	{0xFE50, 0xFE6F, "SmallFormVariants"},
	{0xFE70, 0xFEFE, "ArabicPresentationForms-B"},
	{0xFEFF, 0xFEFF, "Specials"},
	{0xFF00, 0xFFEF, "HalfwidthandFullwidthForms"},
	{0xFFF0, 0xFFFD, "Specials"},
	{0, 0, NULL}
};

typedef struct CategoryCharRangeTag
{
	FLMUNICODE	uzLowChar;
	FLMUNICODE	uzHighChar;
} CATEGORY_CHAR_RANGE;

CATEGORY_CHAR_RANGE LuRanges[] =
{
	{0x0041, 0x005A},
	{0x00C0, 0x00DE},
	{0x0100, 0x0000},
	{0x0102, 0x0000},
	{0x0104, 0x0000},
	{0x0106, 0x0000},
	{0x0108, 0x0000},
	{0x010A, 0x0000},
	{0x010C, 0x0000},
	{0x010E, 0x0000},
	{0x0110, 0x0000},
	{0x0112, 0x0000},
	{0x0114, 0x0000},
	{0x0116, 0x0000},
	{0x0118, 0x0000},
	{0x011A, 0x0000},
	{0x011C, 0x0000},
	{0x011E, 0x0000},
	{0x0120, 0x0000},
	{0x0122, 0x0000},
	{0x0124, 0x0000},
	{0x0126, 0x0000},
	{0x0128, 0x0000},
	{0x012A, 0x0000},
	{0x012C, 0x0000},
	{0x012E, 0x0000},
	{0x0130, 0x0000},
	{0x0132, 0x0000},
	{0x0134, 0x0000},
	{0x0136, 0x0000},
	{0x0139, 0x0000},
	{0x013B, 0x0000},
	{0x013D, 0x0000},
	{0x013F, 0x0000},
	{0x0141, 0x0000},
	{0x0143, 0x0000},
	{0x0145, 0x0000},
	{0x0147, 0x0000},
	{0x014A, 0x0000},
	{0x014C, 0x0000},
	{0x014E, 0x0000},
	{0x0150, 0x0000},
	{0x0152, 0x0000},
	{0x0154, 0x0000},
	{0x0156, 0x0000},
	{0x0158, 0x0000},
	{0x015A, 0x0000},
	{0x015C, 0x0000},
	{0x015E, 0x0000},
	{0x0160, 0x0000},
	{0x0162, 0x0000},
	{0x0164, 0x0000},
	{0x0166, 0x0000},
	{0x0168, 0x0000},
	{0x016A, 0x0000},
	{0x016C, 0x0000},
	{0x016E, 0x0000},
	{0x0170, 0x0000},
	{0x0172, 0x0000},
	{0x0174, 0x0000},
	{0x0176, 0x0000},
	{0x0178, 0x0179},
	{0x017B, 0x0000},
	{0x017D, 0x0000},
	{0x0181, 0x0182},
	{0x0184, 0x0000},
	{0x0186, 0x0187},
	{0x0189, 0x018B},
	{0x018E, 0x0191},
	{0x0193, 0x0194},
	{0x0196, 0x0198},
	{0x019C, 0x019D},
	{0x019F, 0x01A0},
	{0x01A2, 0x0000},
	{0x01A4, 0x0000},
	{0x01A6, 0x01A7},
	{0x01A9, 0x01AC},
	{0x01AE, 0x01AF},
	{0x01B1, 0x01B3},
	{0x01B5, 0x0000},
	{0x01B7, 0x01B8},
	{0x01BC, 0x0000},
	{0x01C4, 0x0000},
	{0x01C7, 0x0000},
	{0x01CA, 0x0000},
	{0x01CD, 0x0000},
	{0x01CF, 0x0000},
	{0x01D1, 0x0000},
	{0x01D3, 0x0000},
	{0x01D5, 0x0000},
	{0x01D7, 0x0000},
	{0x01D9, 0x0000},
	{0x01DB, 0x0000},
	{0x01DE, 0x0000},
	{0x01E0, 0x0000},
	{0x01E2, 0x0000},
	{0x01E4, 0x0000},
	{0x01E6, 0x0000},
	{0x01E8, 0x0000},
	{0x01EA, 0x0000},
	{0x01EC, 0x0000},
	{0x01EE, 0x0000},
	{0x01F1, 0x0000},
	{0x01F4, 0x0000},
	{0x01F6, 0x01F8},
	{0x01FA, 0x0000},
	{0x01FC, 0x0000},
	{0x01FE, 0x0000},
	{0x0200, 0x0000},
	{0x0202, 0x0000},
	{0x0204, 0x0000},
	{0x0206, 0x0000},
	{0x0208, 0x0000},
	{0x020A, 0x0000},
	{0x020C, 0x0000},
	{0x020E, 0x0000},
	{0x0210, 0x0000},
	{0x0212, 0x0000},
	{0x0214, 0x0000},
	{0x0216, 0x0000},
	{0x0218, 0x0000},
	{0x021A, 0x0000},
	{0x021C, 0x0000},
	{0x021E, 0x0000},
	{0x0220, 0x0000},
	{0x0222, 0x0000},
	{0x0224, 0x0000},
	{0x0226, 0x0000},
	{0x0228, 0x0000},
	{0x022A, 0x0000},
	{0x022C, 0x0000},
	{0x022E, 0x0000},
	{0x0230, 0x0000},
	{0x0232, 0x0000},
	{0x0386, 0x0000},
	{0x0388, 0x038A},
	{0x038C, 0x0000},
	{0x038E, 0x038F},
	{0x0391, 0x03AB},
	{0x03D2, 0x03D4},
	{0x03D8, 0x0000},
	{0x03DA, 0x0000},
	{0x03DC, 0x0000},
	{0x03DE, 0x0000},
	{0x03E0, 0x0000},
	{0x03E2, 0x0000},
	{0x03E4, 0x0000},
	{0x03E6, 0x0000},
	{0x03E8, 0x0000},
	{0x03EA, 0x0000},
	{0x03EC, 0x0000},
	{0x03EE, 0x0000},
	{0x03F4, 0x0000},
	{0x03F7, 0x0000},
	{0x03F9, 0x0000},
	{0x03FA, 0x0000},
	{0x0400, 0x042F},
	{0x0460, 0x0000},
	{0x0462, 0x0000},
	{0x0464, 0x0000},
	{0x0466, 0x0000},
	{0x0468, 0x0000},
	{0x046A, 0x0000},
	{0x046C, 0x0000},
	{0x046E, 0x0000},
	{0x0470, 0x0000},
	{0x0472, 0x0000},
	{0x0474, 0x0000},
	{0x0476, 0x0000},
	{0x0478, 0x0000},
	{0x047A, 0x0000},
	{0x047C, 0x0000},
	{0x047E, 0x0000},
	{0x0480, 0x0000},
	{0x048A, 0x0000},
	{0x048C, 0x0000},
	{0x048E, 0x0000},
	{0x0490, 0x0000},
	{0x0492, 0x0000},
	{0x0494, 0x0000},
	{0x0496, 0x0000},
	{0x0498, 0x0000},
	{0x049A, 0x0000},
	{0x049C, 0x0000},
	{0x049E, 0x0000},
	{0x04A0, 0x0000},
	{0x04A2, 0x0000},
	{0x04A4, 0x0000},
	{0x04A6, 0x0000},
	{0x04A8, 0x0000},
	{0x04AA, 0x0000},
	{0x04AC, 0x0000},
	{0x04AE, 0x0000},
	{0x04B0, 0x0000},
	{0x04B2, 0x0000},
	{0x04B4, 0x0000},
	{0x04B6, 0x0000},
	{0x04B8, 0x0000},
	{0x04BA, 0x0000},
	{0x04BC, 0x0000},
	{0x04BE, 0x0000},
	{0x04C0, 0x04C1},
	{0x04C3, 0x0000},
	{0x04C5, 0x0000},
	{0x04C7, 0x0000},
	{0x04C9, 0x0000},
	{0x04CB, 0x0000},
	{0x04CD, 0x0000},
	{0x04D0, 0x0000},
	{0x04D2, 0x0000},
	{0x04D4, 0x0000},
	{0x04D6, 0x0000},
	{0x04D8, 0x0000},
	{0x04DA, 0x0000},
	{0x04DC, 0x0000},
	{0x04DE, 0x0000},
	{0x04E0, 0x0000},
	{0x04E2, 0x0000},
	{0x04E4, 0x0000},
	{0x04E6, 0x0000},
	{0x04E8, 0x0000},
	{0x04EA, 0x0000},
	{0x04EC, 0x0000},
	{0x04EE, 0x0000},
	{0x04F0, 0x0000},
	{0x04F2, 0x0000},
	{0x04F4, 0x0000},
	{0x04F8, 0x0000},
	{0x0500, 0x0000},
	{0x0502, 0x0000},
	{0x0504, 0x0000},
	{0x0506, 0x0000},
	{0x0508, 0x0000},
	{0x050A, 0x0000},
	{0x050C, 0x0000},
	{0x050E, 0x0000},
	{0x0531, 0x0556},
	{0x10A0, 0x10C5},
	{0x1E00, 0x0000},
	{0x1E02, 0x0000},
	{0x1E04, 0x0000},
	{0x1E06, 0x0000},
	{0x1E08, 0x0000},
	{0x1E0A, 0x0000},
	{0x1E0C, 0x0000},
	{0x1E0E, 0x0000},
	{0x1E10, 0x0000},
	{0x1E12, 0x0000},
	{0x1E14, 0x0000},
	{0x1E16, 0x0000},
	{0x1E18, 0x0000},
	{0x1E1A, 0x0000},
	{0x1E1C, 0x0000},
	{0x1E1E, 0x0000},
	{0x1E20, 0x0000},
	{0x1E22, 0x0000},
	{0x1E24, 0x0000},
	{0x1E26, 0x0000},
	{0x1E28, 0x0000},
	{0x1E2A, 0x0000},
	{0x1E2C, 0x0000},
	{0x1E2E, 0x0000},
	{0x1E30, 0x0000},
	{0x1E32, 0x0000},
	{0x1E34, 0x0000},
	{0x1E36, 0x0000},
	{0x1E38, 0x0000},
	{0x1E3A, 0x0000},
	{0x1E3C, 0x0000},
	{0x1E3E, 0x0000},
	{0x1E40, 0x0000},
	{0x1E42, 0x0000},
	{0x1E44, 0x0000},
	{0x1E46, 0x0000},
	{0x1E48, 0x0000},
	{0x1E4A, 0x0000},
	{0x1E4C, 0x0000},
	{0x1E4E, 0x0000},
	{0x1E50, 0x0000},
	{0x1E52, 0x0000},
	{0x1E54, 0x0000},
	{0x1E56, 0x0000},
	{0x1E58, 0x0000},
	{0x1E5A, 0x0000},
	{0x1E5C, 0x0000},
	{0x1E5E, 0x0000},
	{0x1E60, 0x0000},
	{0x1E62, 0x0000},
	{0x1E64, 0x0000},
	{0x1E66, 0x0000},
	{0x1E68, 0x0000},
	{0x1E6A, 0x0000},
	{0x1E6C, 0x0000},
	{0x1E6E, 0x0000},
	{0x1E70, 0x0000},
	{0x1E72, 0x0000},
	{0x1E74, 0x0000},
	{0x1E76, 0x0000},
	{0x1E78, 0x0000},
	{0x1E7A, 0x0000},
	{0x1E7C, 0x0000},
	{0x1E7E, 0x0000},
	{0x1E80, 0x0000},
	{0x1E82, 0x0000},
	{0x1E84, 0x0000},
	{0x1E86, 0x0000},
	{0x1E88, 0x0000},
	{0x1E8A, 0x0000},
	{0x1E8C, 0x0000},
	{0x1E8E, 0x0000},
	{0x1E90, 0x0000},
	{0x1E92, 0x0000},
	{0x1E94, 0x0000},
	{0x1EA0, 0x0000},
	{0x1EA2, 0x0000},
	{0x1EA4, 0x0000},
	{0x1EA6, 0x0000},
	{0x1EA8, 0x0000},
	{0x1EAA, 0x0000},
	{0x1EAC, 0x0000},
	{0x1EAE, 0x0000},
	{0x1EB0, 0x0000},
	{0x1EB2, 0x0000},
	{0x1EB4, 0x0000},
	{0x1EB6, 0x0000},
	{0x1EB8, 0x0000},
	{0x1EBA, 0x0000},
	{0x1EBC, 0x0000},
	{0x1EBE, 0x0000},
	{0x1EC0, 0x0000},
	{0x1EC2, 0x0000},
	{0x1EC4, 0x0000},
	{0x1EC6, 0x0000},
	{0x1EC8, 0x0000},
	{0x1ECA, 0x0000},
	{0x1ECC, 0x0000},
	{0x1ECE, 0x0000},
	{0x1ED0, 0x0000},
	{0x1ED2, 0x0000},
	{0x1ED4, 0x0000},
	{0x1ED6, 0x0000},
	{0x1ED8, 0x0000},
	{0x1EDA, 0x0000},
	{0x1EDC, 0x0000},
	{0x1EDE, 0x0000},
	{0x1EE0, 0x0000},
	{0x1EE2, 0x0000},
	{0x1EE4, 0x0000},
	{0x1EE6, 0x0000},
	{0x1EE8, 0x0000},
	{0x1EEA, 0x0000},
	{0x1EEC, 0x0000},
	{0x1EEE, 0x0000},
	{0x1EF0, 0x0000},
	{0x1EF2, 0x0000},
	{0x1EF4, 0x0000},
	{0x1EF6, 0x0000},
	{0x1EF8, 0x0000},
	{0x1F08, 0x1F0F},
	{0x1F18, 0x1F1D},
	{0x1F28, 0x1F2F},
	{0x1F38, 0x1F3F},
	{0x1F49, 0x1F4D},
	{0x1F59, 0x0000},
	{0x1F5B, 0x0000},
	{0x1F5D, 0x0000},
	{0x1F5F, 0x0000},
	{0x1F68, 0x1F6F},
	{0x1FB8, 0x1FBB},
	{0x1FC8, 0x1FCB},
	{0x1FD8, 0x1FDB},
	{0x1FE8, 0x1FEC},
	{0x1FF8, 0x1FFB},
	{0x2102, 0x0000},
	{0x2107, 0x0000},
	{0x210B, 0x210D},
	{0x2110, 0x2112},
	{0x2115, 0x0000},
	{0x2119, 0x211D},
	{0x2124, 0x0000},
	{0x2126, 0x0000},
	{0x2128, 0x0000},
	{0x212A, 0x212D},
	{0x2130, 0x2131},
	{0x2133, 0x0000},
	{0x213E, 0x213F},
	{0x2145, 0x0000},
	{0xFF21, 0xFF3A}
};

CATEGORY_CHAR_RANGE LlRanges[] =
{
	{0x0061, 0x007A},
	{0x00AA, 0x0000},
	{0x00B5, 0x0000},
	{0x00BA, 0x0000},
	{0x00DF, 0x0000},
	{0x00E0, 0x00FF},
	{0x0101, 0x0000},
	{0x0103, 0x0000},
	{0x0105, 0x0000},
	{0x0107, 0x0000},
	{0x0109, 0x0000},
	{0x010B, 0x0000},
	{0x010D, 0x0000},
	{0x010F, 0x0000},
	{0x0111, 0x0000},
	{0x0113, 0x0000},
	{0x0115, 0x0000},
	{0x0117, 0x0000},
	{0x0119, 0x0000},
	{0x011B, 0x0000},
	{0x011D, 0x0000},
	{0x011F, 0x0000},
	{0x0121, 0x0000},
	{0x0123, 0x0000},
	{0x0125, 0x0000},
	{0x0127, 0x0000},
	{0x0129, 0x0000},
	{0x012B, 0x0000},
	{0x012D, 0x0000},
	{0x012F, 0x0000},
	{0x0131, 0x0000},
	{0x0133, 0x0000},
	{0x0135, 0x0000},
	{0x0137, 0x0138},
	{0x013A, 0x0000},
	{0x013C, 0x0000},
	{0x013E, 0x0000},
	{0x0140, 0x0000},
	{0x0142, 0x0000},
	{0x0144, 0x0000},
	{0x0146, 0x0000},
	{0x0148, 0x0149},
	{0x014B, 0x0000},
	{0x014D, 0x0000},
	{0x014F, 0x0000},
	{0x0151, 0x0000},
	{0x0153, 0x0000},
	{0x0155, 0x0000},
	{0x0157, 0x0000},
	{0x0159, 0x0000},
	{0x015B, 0x0000},
	{0x015D, 0x0000},
	{0x015F, 0x0000},
	{0x0161, 0x0000},
	{0x0163, 0x0000},
	{0x0165, 0x0000},
	{0x0167, 0x0000},
	{0x0169, 0x0000},
	{0x016B, 0x0000},
	{0x016D, 0x0000},
	{0x016F, 0x0000},
	{0x0171, 0x0000},
	{0x0173, 0x0000},
	{0x0175, 0x0000},
	{0x0177, 0x0000},
	{0x017A, 0x0000},
	{0x017C, 0x0000},
	{0x017E, 0x017F},
	{0x0180, 0x0000},
	{0x0183, 0x0000},
	{0x0185, 0x0000},
	{0x0188, 0x0000},
	{0x018C, 0x018D},
	{0x0192, 0x0000},
	{0x0195, 0x0000},
	{0x0199, 0x019B},
	{0x019E, 0x0000},
	{0x01A1, 0x0000},
	{0x01A3, 0x0000},
	{0x01A5, 0x0000},
	{0x01A8, 0x0000},
	{0x01AA, 0x01AB},
	{0x01AD, 0x0000},
	{0x01B0, 0x0000},
	{0x01B4, 0x0000},
	{0x01B6, 0x0000},
	{0x01B9, 0x01BA},
	{0x01BD, 0x01BF},
	{0x01C6, 0x0000},
	{0x01C9, 0x0000},
	{0x01CC, 0x0000},
	{0x01CE, 0x0000},
	{0x01D2, 0x0000},
	{0x01D4, 0x0000},
	{0x01D6, 0x0000},
	{0x01D8, 0x0000},
	{0x01DA, 0x0000},
	{0x01DC, 0x01DD},
	{0x01DF, 0x0000},
	{0x01E1, 0x0000},
	{0x01E3, 0x0000},
	{0x01E5, 0x0000},
	{0x01E7, 0x0000},
	{0x01E9, 0x0000},
	{0x01EB, 0x0000},
	{0x01ED, 0x0000},
	{0x01EF, 0x0000},
	{0x01F0, 0x0000},
	{0x01F3, 0x0000},
	{0x01F5, 0x0000},
	{0x01F9, 0x0000},
	{0x01FB, 0x0000},
	{0x01FD, 0x0000},
	{0x01FF, 0x0000},
	{0x0201, 0x0000},
	{0x0203, 0x0000},
	{0x0205, 0x0000},
	{0x0207, 0x0000},
	{0x0209, 0x0000},
	{0x020B, 0x0000},
	{0x020D, 0x0000},
	{0x020F, 0x0000},
	{0x0211, 0x0000},
	{0x0213, 0x0000},
	{0x0215, 0x0000},
	{0x0217, 0x0000},
	{0x0219, 0x0000},
	{0x021B, 0x0000},
	{0x021D, 0x0000},
	{0x021F, 0x0000},
	{0x0221, 0x0000},
	{0x0223, 0x0000},
	{0x0225, 0x0000},
	{0x0227, 0x0000},
	{0x0229, 0x0000},
	{0x022B, 0x0000},
	{0x022D, 0x0000},
	{0x022F, 0x0000},
	{0x0231, 0x0000},
	{0x0233, 0x0236},
	{0x0234, 0x0000},
	{0x0250, 0x0000},
	{0x0209, 0x0000},
	{0x020B, 0x0000},
	{0x020D, 0x0000},
	{0x020F, 0x02AF},
	{0x0390, 0x0000},
	{0x03AC, 0x03CE},
	{0x03D0, 0x03D1},
	{0x03D5, 0x03D7},
	{0x03D9, 0x0000},
	{0x03DB, 0x0000},
	{0x03DD, 0x0000},
	{0x03DF, 0x0000},
	{0x03E1, 0x0000},
	{0x03E3, 0x0000},
	{0x03E5, 0x0000},
	{0x03E7, 0x0000},
	{0x03E9, 0x0000},
	{0x03EB, 0x0000},
	{0x03ED, 0x0000},
	{0x03EF, 0x03F3},
	{0x03F5, 0x0000},
	{0x03F8, 0x0000},
	{0x03FB, 0x0000},
	{0x0430, 0x045F},
	{0x0461, 0x0000},
	{0x0463, 0x0000},
	{0x0465, 0x0000},
	{0x0467, 0x0000},
	{0x0469, 0x0000},
	{0x046B, 0x0000},
	{0x046D, 0x0000},
	{0x046F, 0x0000},
	{0x0471, 0x0000},
	{0x0473, 0x0000},
	{0x0475, 0x0000},
	{0x0477, 0x0000},
	{0x0479, 0x0000},
	{0x047B, 0x0000},
	{0x047D, 0x0000},
	{0x047F, 0x0000},
	{0x0481, 0x0000},
	{0x048B, 0x0000},
	{0x048D, 0x0000},
	{0x048F, 0x0000},
	{0x0491, 0x0000},
	{0x0493, 0x0000},
	{0x0495, 0x0000},
	{0x0497, 0x0000},
	{0x0499, 0x0000},
	{0x049B, 0x0000},
	{0x049D, 0x0000},
	{0x049F, 0x0000},
	{0x04A1, 0x0000},
	{0x04A3, 0x0000},
	{0x04A5, 0x0000},
	{0x04A7, 0x0000},
	{0x04A9, 0x0000},
	{0x04AB, 0x0000},
	{0x04AD, 0x0000},
	{0x04AF, 0x0000},
	{0x04B1, 0x0000},
	{0x04B3, 0x0000},
	{0x04B5, 0x0000},
	{0x04B7, 0x0000},
	{0x04B9, 0x0000},
	{0x04BB, 0x0000},
	{0x04BD, 0x0000},
	{0x04BF, 0x0000},
	{0x04C2, 0x0000},
	{0x04C4, 0x0000},
	{0x04C6, 0x0000},
	{0x04C8, 0x0000},
	{0x04CA, 0x0000},
	{0x04CC, 0x0000},
	{0x04CE, 0x0000},
	{0x04D1, 0x0000},
	{0x04D3, 0x0000},
	{0x04D5, 0x0000},
	{0x04D7, 0x0000},
	{0x04D9, 0x0000},
	{0x04DB, 0x0000},
	{0x04DD, 0x0000},
	{0x04DF, 0x0000},
	{0x04E1, 0x0000},
	{0x04E3, 0x0000},
	{0x04E5, 0x0000},
	{0x04E7, 0x0000},
	{0x04E9, 0x0000},
	{0x04EB, 0x0000},
	{0x04ED, 0x0000},
	{0x04EF, 0x0000},
	{0x04F1, 0x0000},
	{0x04F3, 0x0000},
	{0x04F5, 0x0000},
	{0x04F9, 0x0000},
	{0x0501, 0x0000},
	{0x0503, 0x0000},
	{0x0505, 0x0000},
	{0x0507, 0x0000},
	{0x0509, 0x0000},
	{0x050B, 0x0000},
	{0x050D, 0x0000},
	{0x050F, 0x0000},
	{0x0561, 0x0587},
	{0x1D00, 0x1D2B},
	{0x1D62, 0x1D6B},
	{0x1E01, 0x0000},
	{0x1E03, 0x0000},
	{0x1E05, 0x0000},
	{0x1E07, 0x0000},
	{0x1E09, 0x0000},
	{0x1E0B, 0x0000},
	{0x1E0D, 0x0000},
	{0x1E0F, 0x0000},
	{0x1E11, 0x0000},
	{0x1E13, 0x0000},
	{0x1E15, 0x0000},
	{0x1E17, 0x0000},
	{0x1E19, 0x0000},
	{0x1E1B, 0x0000},
	{0x1E1D, 0x0000},
	{0x1E1F, 0x0000},
	{0x1E21, 0x0000},
	{0x1E23, 0x0000},
	{0x1E25, 0x0000},
	{0x1E27, 0x0000},
	{0x1E29, 0x0000},
	{0x1E2B, 0x0000},
	{0x1E2D, 0x0000},
	{0x1E2F, 0x0000},
	{0x1E31, 0x0000},
	{0x1E33, 0x0000},
	{0x1E35, 0x0000},
	{0x1E37, 0x0000},
	{0x1E39, 0x0000},
	{0x1E3B, 0x0000},
	{0x1E3D, 0x0000},
	{0x1E3F, 0x0000},
	{0x1E41, 0x0000},
	{0x1E43, 0x0000},
	{0x1E45, 0x0000},
	{0x1E47, 0x0000},
	{0x1E49, 0x0000},
	{0x1E4B, 0x0000},
	{0x1E4D, 0x0000},
	{0x1E4F, 0x0000},
	{0x1E51, 0x0000},
	{0x1E53, 0x0000},
	{0x1E55, 0x0000},
	{0x1E57, 0x0000},
	{0x1E59, 0x0000},
	{0x1E5B, 0x0000},
	{0x1E5D, 0x0000},
	{0x1E5F, 0x0000},
	{0x1E61, 0x0000},
	{0x1E63, 0x0000},
	{0x1E65, 0x0000},
	{0x1E67, 0x0000},
	{0x1E69, 0x0000},
	{0x1E6B, 0x0000},
	{0x1E6D, 0x0000},
	{0x1E6F, 0x0000},
	{0x1E71, 0x0000},
	{0x1E73, 0x0000},
	{0x1E75, 0x0000},
	{0x1E77, 0x0000},
	{0x1E79, 0x0000},
	{0x1E7B, 0x0000},
	{0x1E7D, 0x0000},
	{0x1E7F, 0x0000},
	{0x1E81, 0x0000},
	{0x1E83, 0x0000},
	{0x1E85, 0x0000},
	{0x1E87, 0x0000},
	{0x1E89, 0x0000},
	{0x1E8B, 0x0000},
	{0x1E8D, 0x0000},
	{0x1E8F, 0x0000},
	{0x1E91, 0x0000},
	{0x1E93, 0x0000},
	{0x1E95, 0x1E9B},
	{0x1EA1, 0x0000},
	{0x1EA3, 0x0000},
	{0x1EA5, 0x0000},
	{0x1EA7, 0x0000},
	{0x1EA9, 0x0000},
	{0x1EAB, 0x0000},
	{0x1EAD, 0x0000},
	{0x1EAF, 0x0000},
	{0x1EB1, 0x0000},
	{0x1EB3, 0x0000},
	{0x1EB5, 0x0000},
	{0x1EB7, 0x0000},
	{0x1EB9, 0x0000},
	{0x1EBB, 0x0000},
	{0x1EBD, 0x0000},
	{0x1EBF, 0x0000},
	{0x1EC1, 0x0000},
	{0x1EC3, 0x0000},
	{0x1EC5, 0x0000},
	{0x1EC7, 0x0000},
	{0x1EC9, 0x0000},
	{0x1ECB, 0x0000},
	{0x1ECD, 0x0000},
	{0x1ECF, 0x0000},
	{0x1ED1, 0x0000},
	{0x1ED3, 0x0000},
	{0x1ED5, 0x0000},
	{0x1ED7, 0x0000},
	{0x1ED9, 0x0000},
	{0x1EDB, 0x0000},
	{0x1EDD, 0x0000},
	{0x1EDF, 0x0000},
	{0x1EE1, 0x0000},
	{0x1EE3, 0x0000},
	{0x1EE5, 0x0000},
	{0x1EE7, 0x0000},
	{0x1EE9, 0x0000},
	{0x1EEB, 0x0000},
	{0x1EED, 0x0000},
	{0x1EEF, 0x0000},
	{0x1EF1, 0x0000},
	{0x1EF3, 0x0000},
	{0x1EF5, 0x0000},
	{0x1EF7, 0x0000},
	{0x1EF9, 0x0000},
	{0x1F00, 0x1F07},
	{0x1F10, 0x1F15},
	{0x1F20, 0x1F27},
	{0x1F30, 0x1F37},
	{0x1F40, 0x1F45},
	{0x1F50, 0x1F57},
	{0x1F60, 0x1F67},
	{0x1F70, 0x1F7D},
	{0x1F80, 0x1F87},
	{0x1F90, 0x1F97},
	{0x1FA0, 0x1FA7},
	{0x1FB0, 0x1FB7},
	{0x1FBE, 0x0000},
	{0x1FC2, 0x1FC4},
	{0x1FC6, 0x1FC7},
	{0x1FD0, 0x1FD7},
	{0x1FE0, 0x1FE7},
	{0x1FF2, 0x1FF4},
	{0x1FF6, 0x1FF7},
	{0x2071, 0x0000},
	{0x207F, 0x0000},
	{0x210A, 0x0000},
	{0x210E, 0x210F},
	{0x2113, 0x0000},
	{0x212F, 0x0000},
	{0x2134, 0x0000},
	{0x2139, 0x0000},
	{0x213D, 0x0000},
	{0x2146, 0x2149},
	{0xFB00, 0xFB06},
	{0xFB13, 0xFB17},
	{0xFF41, 0xFF5A},
	{0x1E0B, 0x0000},
	{0x1E0D, 0x0000},
	{0x1E0F, 0x0000}
};

CATEGORY_CHAR_RANGE LtRanges[] =
{
	{0x01C5, 0x0000},
	{0x01C8, 0x0000},
	{0x01CB, 0x0000},
	{0x01F2, 0x0000},
	{0x1F88, 0x1F8F},
	{0x1F98, 0x1F9F},
	{0x1FA8, 0x1FAF},
	{0x1FBC, 0x0000},
	{0x1FCC, 0x0000},
	{0x1FFC, 0x0000}
};

CATEGORY_CHAR_RANGE LmRanges[] =
{
	{0x02B0, 0x02C1},
	{0x02C6, 0x02D1},
	{0x02E0, 0x02E4},
	{0x02EE, 0x0000},
	{0x037A, 0x0000},
	{0x0559, 0x0000},
	{0x0640, 0x0000},
	{0x06E5, 0x06E6},
	{0x0E46, 0x0000},
	{0x0EC6, 0x0000},
	{0x17D7, 0x0000},
	{0x1843, 0x0000},
	{0x1D2C, 0x1D61},
	{0x3005, 0x0000},
	{0x3031, 0x3035},
	{0x303B, 0x0000},
	{0x309D, 0x309E},
	{0x30FC, 0x30FE},
	{0xFF70, 0x0000},
	{0xFF9E, 0xFF9F}
};

CATEGORY_CHAR_RANGE LoRanges[] =
{
	{0x01BB, 0x0000},
	{0x01C0, 0x01C3},
	{0x05D0, 0x05EA},
	{0x05F0, 0x05F2},
	{0x0621, 0x063A},
	{0x0641, 0x064A},
	{0x066E, 0x066F},
	{0x0671, 0x06D3},
	{0x06D5, 0x0000},
	{0x06EE, 0x06EF},
	{0x06FA, 0x06FC},
	{0x06FF, 0x0000},
	{0x0710, 0x0000},
	{0x0712, 0x072F},
	{0x074D, 0x074F},
	{0x0780, 0x07A5},
	{0x07B1, 0x0000},
	{0x0904, 0x0939},
	{0x093D, 0x0000},
	{0x0950, 0x0000},
	{0x0958, 0x0961},
	{0x0985, 0x098C},
	{0x098F, 0x0990},
	{0x0993, 0x09A8},
	{0x09AA, 0x09B0},
	{0x09B2, 0x0000},
	{0x09B6, 0x09B9},
	{0x09BD, 0x0000},
	{0x09DC, 0x09DD},
	{0x09DF, 0x09E1},
	{0x09F0, 0x09F1},
	{0x0A05, 0x0A0A},
	{0x0A0F, 0x0A10},
	{0x0A13, 0x0A28},
	{0x0A2A, 0x0A30},
	{0x0A32, 0x0A33},
	{0x0A35, 0x0A36},
	{0x0A38, 0x0A39},
	{0x0A59, 0x0A5C},
	{0x0A5E, 0x0000},
	{0x0A72, 0x0A74},
	{0x0A85, 0x0A8D},
	{0x0A8F, 0x0A91},
	{0x0A93, 0x0AB0},
	{0x0AB2, 0x0AB3},
	{0x0AB5, 0x0AB9},
	{0x0ABD, 0x0000},
	{0x0AD0, 0x0000},
	{0x0AE0, 0x0AE1},
	{0x0B05, 0x0B0C},
	{0x0B0F, 0x0B10},
	{0x0B13, 0x0B30},
	{0x0B32, 0x0B33},
	{0x0B35, 0x0B39},
	{0x0B3D, 0x0000},
	{0x0B5C, 0x0B5D},
	{0x0B5F, 0x0B61},
	{0x0B71, 0x0000},
	{0x0B83, 0x0000},
	{0x0B85, 0x0B8A},
	{0x0B8E, 0x0B90},
	{0x0B92, 0x0B95},
	{0x0B99, 0x0B9A},
	{0x0B9C, 0x0000},
	{0x0B9E, 0x0B9F},
	{0x0BA3, 0x0BA4},
	{0x0BA8, 0x0BAA},
	{0x0BAE, 0x0BAF},
	{0x0BB0, 0x0BB9},
	{0x0C05, 0x0C0C},
	{0x0C0E, 0x0C10},
	{0x0C12, 0x0C28},
	{0x0C2A, 0x0C33},
	{0x0C35, 0x0C39},
	{0x0C60, 0x0C61},
	{0x0C85, 0x0C8C},
	{0x0C8E, 0x0C90},
	{0x0C92, 0x0CB3},
	{0x0CB5, 0x0CB9},
	{0x0CBD, 0x0000},
	{0x0CDE, 0x0000},
	{0x0CE0, 0x0CE1},
	{0x0D05, 0x0D0C},
	{0x0D0E, 0x0D10},
	{0x0D12, 0x0D28},
	{0x0D2A, 0x0D39},
	{0x0D60, 0x0D61},
	{0x0D85, 0x0D96},
	{0x0D9A, 0x0DB1},
	{0x0DB3, 0x0DBB},
	{0x0DBD, 0x0000},
	{0x0DC0, 0x0DC6},
	{0x0E01, 0x0E30},
	{0x0E32, 0x0E33},
	{0x0E40, 0x0E45},
	{0x0E81, 0x0E82},
	{0x0E84, 0x0000},
	{0x0E87, 0x0E88},
	{0x0E8A, 0x0000},
	{0x0E8D, 0x0000},
	{0x0E94, 0x0E97},
	{0x0E99, 0x0E9F},
	{0x0EA1, 0x0EA3},
	{0x0EA5, 0x0000},
	{0x0EA7, 0x0000},
	{0x0EAA, 0x0EAB},
	{0x0EAD, 0x0EB0},
	{0x0EB2, 0x0EB3},
	{0x0EBD, 0x0000},
	{0x0EC0, 0x0EC4},
	{0x0EDC, 0x0EDD},
	{0x0F00, 0x0000},
	{0x0F40, 0x0F47},
	{0x0F49, 0x0F6A},
	{0x0F88, 0x0F8B},
	{0x1000, 0x1021},
	{0x1023, 0x1027},
	{0x1029, 0x102A},
	{0x1050, 0x1055},
	{0x10D0, 0x10F8},
	{0x1100, 0x11A2},
	{0x11A8, 0x11F9},
	{0x1200, 0x1248},
	{0x124A, 0x124D},
	{0x1250, 0x1256},
	{0x1258, 0x0000},
	{0x125A, 0x125D},
	{0x1260, 0x1286},
	{0x1288, 0x0000},
	{0x128A, 0x128D},
	{0x1290, 0x12AE},
	{0x12B0, 0x0000},
	{0x12B2, 0x12B5},
	{0x12B8, 0x12BE},
	{0x12C0, 0x0000},
	{0x12C2, 0x12C5},
	{0x12C8, 0x12CE},
	{0x12D0, 0x12D6},
	{0x12D8, 0x12EE},
	{0x12F0, 0x130E},
	{0x1310, 0x0000},
	{0x1312, 0x1315},
	{0x1318, 0x131E},
	{0x1320, 0x1346},
	{0x1348, 0x135A},
	{0x13A0, 0x13F4},
	{0x1401, 0x166C},
	{0x166F, 0x1676},
	{0x1681, 0x169A},
	{0x16A0, 0x16EA},
	{0x1700, 0x170C},
	{0x170E, 0x1711},
	{0x1720, 0x1731},
	{0x1740, 0x1751},
	{0x1760, 0x176C},
	{0x176E, 0x1770},
	{0x1780, 0x17B3},
	{0x17DC, 0x0000},
	{0x1820, 0x1842},
	{0x1844, 0x1877},
	{0x1880, 0x18A8},
	{0x1900, 0x191C},
	{0x1950, 0x196D},
	{0x1970, 0x1974},
	{0x2135, 0x2138},
	{0x3006, 0x0000},
	{0x303C, 0x0000},
	{0x3041, 0x3096},
	{0x309F, 0x0000},
	{0x30A1, 0x30FA},
	{0x30FF, 0x0000},
	{0x3105, 0x312C},
	{0x3131, 0x318E},
	{0x31A0, 0x31B7},
	{0x31F0, 0x31FF},
	{0x3400, 0x0000},
	{0x4DB5, 0x0000},
	{0x4E00, 0x0000},
	{0x9FA5, 0x0000},
	{0xA000, 0xA48C},
	{0xAC00, 0x0000},
	{0xD7A3, 0x0000},
	{0xF900, 0xFA2D},
	{0xFA30, 0xFA6A},
	{0xFB1D, 0x0000},
	{0xFB1F, 0xFB28},
	{0xFB2A, 0xFB36},
	{0xFB38, 0xFB3C},
	{0xFB3E, 0x0000},
	{0xFB40, 0xFB41},
	{0xFB43, 0xFB44},
	{0xFB46, 0xFBB1},
	{0xFBD3, 0xFD3D},
	{0xFD50, 0xFD8F},
	{0xFD92, 0xFDC7},
	{0xFDF0, 0xFDFB},
	{0xFE70, 0xFE74},
	{0xFE76, 0xFEFC},
	{0xFF66, 0xFF6F},
	{0xFF71, 0xFF9D},
	{0xFFA0, 0xFFBE},
	{0xFFC2, 0xFFC7},
	{0xFFCA, 0xFFCF},
	{0xFFD2, 0xFFD7},
	{0xFFDA, 0xFFDC}
};

CATEGORY_CHAR_RANGE MnRanges[] =
{
	{0x0300, 0x0357},
	{0x035D, 0x036F},
	{0x0483, 0x0486},
	{0x0591, 0x05A1},
	{0x05A3, 0x05B9},
	{0x05BB, 0x05BD},
	{0x05BF, 0x0000},
	{0x05C1, 0x05C2},
	{0x05C4, 0x0000},
	{0x0610, 0x0615},
	{0x064B, 0x0658},
	{0x0670, 0x0000},
	{0x06D6, 0x06DC},
	{0x06DF, 0x06E4},
	{0x06E7, 0x06E8},
	{0x06EA, 0x06ED},
	{0x0711, 0x0000},
	{0x0730, 0x074A},
	{0x07A6, 0x07B0},
	{0x0901, 0x0902},
	{0x093C, 0x0000},
	{0x0941, 0x0948},
	{0x094D, 0x0000},
	{0x0951, 0x0954},
	{0x0962, 0x0963},
	{0x0981, 0x0000},
	{0x09BC, 0x0000},
	{0x09C1, 0x09C4},
	{0x09CD, 0x0000},
	{0x09E2, 0x09E3},
	{0x0A01, 0x0A02},
	{0x0A3C, 0x0000},
	{0x0A41, 0x0A42},
	{0x0A47, 0x0A48},
	{0x0A4B, 0x0A4D},
	{0x0A70, 0x0A71},
	{0x0A81, 0x0A82},
	{0x0ABC, 0x0000},
	{0x0AC1, 0x0AC5},
	{0x0AC7, 0x0AC8},
	{0x0ACD, 0x0000},
	{0x0AE2, 0x0AE3},
	{0x0B01, 0x0000},
	{0x0B3C, 0x0000},
	{0x0B3F, 0x0000},
	{0x0B41, 0x0B43},
	{0x0B4D, 0x0000},
	{0x0B56, 0x0000},
	{0x0B82, 0x0000},
	{0x0BC0, 0x0000},
	{0x0BCD, 0x0000},
	{0x0C3E, 0x0C40},
	{0x0C46, 0x0C48},
	{0x0C4A, 0x0C4D},
	{0x0C55, 0x0C56},
	{0x0CBC, 0x0000},
	{0x0CBF, 0x0000},
	{0x0CC6, 0x0000},
	{0x0CCC, 0x0CCD},
	{0x0D41, 0x0D43},
	{0x0D4D, 0x0000},
	{0x0DCA, 0x0000},
	{0x0DD2, 0x0DD4},
	{0x0DD6, 0x0000},
	{0x0E31, 0x0000},
	{0x0E34, 0x0E3A},
	{0x0E47, 0x0E4E},
	{0x0EB1, 0x0000},
	{0x0EB4, 0x0EB9},
	{0x0EBB, 0x0EBC},
	{0x0EC8, 0x0ECD},
	{0x0F18, 0x0F19},
	{0x0F35, 0x0000},
	{0x0F37, 0x0000},
	{0x0F39, 0x0000},
	{0x0F71, 0x0F7E},
	{0x0F80, 0x0F84},
	{0x0F86, 0x0F87},
	{0x0F90, 0x0F97},
	{0x0F99, 0x0FBC},
	{0x0FC6, 0x0000},
	{0x102D, 0x1030},
	{0x1032, 0x0000},
	{0x1036, 0x1037},
	{0x1039, 0x0000},
	{0x1058, 0x1059},
	{0x1712, 0x1714},
	{0x1732, 0x1734},
	{0x1752, 0x1753},
	{0x1772, 0x1773},
	{0x17B7, 0x17BD},
	{0x17C6, 0x0000},
	{0x17C9, 0x17D3},
	{0x17DD, 0x0000},
	{0x180B, 0x180D},
	{0x18A9, 0x0000},
	{0x1920, 0x1922},
	{0x1927, 0x1928},
	{0x1932, 0x0000},
	{0x1939, 0x193B},
	{0x20D0, 0x20DC},
	{0x20E1, 0x0000},
	{0x20E5, 0x20EA},
	{0x302A, 0x302F},
	{0x3099, 0x309A},
	{0xFB1E, 0x0000},
	{0xFE00, 0xFE0F},
	{0xFE20, 0xFE23}
};

CATEGORY_CHAR_RANGE McRanges[] =
{
	{0x0903, 0x0000},
	{0x093E, 0x0940},
	{0x0949, 0x094C},
	{0x0982, 0x0983},
	{0x09BE, 0x09C0},
	{0x09C7, 0x09C8},
	{0x09CB, 0x09CC},
	{0x09D7, 0x0000},
	{0x0A03, 0x0000},
	{0x0A3E, 0x0A40},
	{0x0A83, 0x0000},
	{0x0ABE, 0x0AC0},
	{0x0AC9, 0x0000},
	{0x0ACB, 0x0ACC},
	{0x0B02, 0x0B03},
	{0x0BE3, 0x0000},
	{0x0B40, 0x0000},
	{0x0B47, 0x0B48},
	{0x0B4B, 0x0B4C},
	{0x0B57, 0x0000},
	{0x0BBE, 0x0BBF},
	{0x0BC1, 0x0BC2},
	{0x0BC6, 0x0BC8},
	{0x0BCA, 0x0BCC},
	{0x0BD7, 0x0000},
	{0x0C01, 0x0C03},
	{0x0C41, 0x0C44},
	{0x0C82, 0x0C83},
	{0x0CBE, 0x0000},
	{0x0CC0, 0x0CC4},
	{0x0CC7, 0x0CC8},
	{0x0CCA, 0x0CCB},
	{0x0CD5, 0x0CD6},
	{0x0D02, 0x0D03},
	{0x0D3E, 0x0D40},
	{0x0D46, 0x0D48},
	{0x0D4A, 0x0D4C},
	{0x0D57, 0x0000},
	{0x0D82, 0x0D83},
	{0x0DCF, 0x0DD1},
	{0x0DD8, 0x0DDF},
	{0x0DF2, 0x0DF3},
	{0x0F3E, 0x0F3F},
	{0x0F7F, 0x0000},
	{0x102C, 0x0000},
	{0x1031, 0x0000},
	{0x1038, 0x0000},
	{0x1056, 0x1057},
	{0x17B6, 0x0000},
	{0x17BE, 0x17C5},
	{0x17C7, 0x17C8},
	{0x1923, 0x1926},
	{0x1929, 0x192B},
	{0x1930, 0x1931},
	{0x1933, 0x1938}
};

CATEGORY_CHAR_RANGE MeRanges[] =
{
	{0x0488, 0x0489},
	{0x06DE, 0x0000},
	{0x20DD, 0x20DF},
	{0x20E0, 0x0000},
	{0x20E2, 0x20E4}
};

CATEGORY_CHAR_RANGE DigitRanges[] =	// Part 1 of Nd category
{
	{0x0030, 0x0039},
	{0x0660, 0x0669},
	{0x06F0, 0x06F9},
	{0x0966, 0x096F},
	{0x09E6, 0x09EF},
	{0x0A66, 0x0A6F},
	{0x0AE6, 0x0AEF},
	{0x0B66, 0x0B6F},
	{0x0BE7, 0x0BEF},
	{0x0C66, 0x0C6F},
	{0x0CE6, 0x0CEF},
	{0x0D66, 0x0D6F},
	{0x0E50, 0x0E59},
	{0x0ED0, 0x0ED9},
	{0x0F20, 0x0F29}
};

CATEGORY_CHAR_RANGE Nd2Ranges[] =	// Part 2 of Nd category
{
	{0x1040, 0x1049},
	{0x1369, 0x1371},
	{0x17E0, 0x17E9},
	{0x1810, 0x1819},
	{0x1946, 0x194F},
	{0xFF10, 0xFF19}
};

CATEGORY_CHAR_RANGE NlRanges[] =
{
	{0x16EE, 0x16F0},
	{0x2160, 0x2183},
	{0x3007, 0x0000},
	{0x3021, 0x3029},
	{0x3038, 0x303A}
};

CATEGORY_CHAR_RANGE NoRanges[] =
{
	{0x00B2, 0x00B3},
	{0x00B9, 0x0000},
	{0x00BC, 0x00BE},
	{0x09F4, 0x09F9},
	{0x0BF0, 0x0BF2},
	{0x0F2A, 0x0F33},
	{0x1372, 0x137C},
	{0x17F0, 0x17F9},
	{0x2070, 0x0000},
	{0x2074, 0x2079},
	{0x2080, 0x2089},
	{0x2153, 0x215F},
	{0x2460, 0x249B},
	{0x24EA, 0x24FF},
	{0x2776, 0x2793},
	{0x3192, 0x3195},
	{0x3220, 0x3229},
	{0x3251, 0x325F},
	{0x3280, 0x3289},
	{0x32B1, 0x32BF}
};

CATEGORY_CHAR_RANGE PcRanges[] =
{
	{0x005F, 0x0000},
	{0x203F, 0x2040},
	{0x2054, 0x0000},
	{0x30FB, 0x0000},
	{0xFE33, 0xFE34},
	{0xFE4D, 0xFE4F},
	{0xFF3F, 0x0000},
	{0xFF65, 0x0000}
};

CATEGORY_CHAR_RANGE PdRanges[] =
{
	{0x002D, 0x0000},
	{0x058A, 0x0000},
	{0x1806, 0x0000},
	{0x2010, 0x2015},
	{0x301C, 0x0000},
	{0x3030, 0x0000},
	{0x30A0, 0x0000},
	{0xFE31, 0xFE32},
	{0xFE58, 0x0000},
	{0xFE63, 0x0000},
	{0xFF0D, 0x0000}
};

CATEGORY_CHAR_RANGE PsRanges[] =
{
	{0x0028, 0x0000},
	{0x005B, 0x0000},
	{0x007B, 0x0000},
	{0x0F3A, 0x0000},
	{0x0F3C, 0x0000},
	{0x169B, 0x0000},
	{0x201A, 0x0000},
	{0x201E, 0x0000},
	{0x2045, 0x0000},
	{0x207D, 0x0000},
	{0x208D, 0x0000},
	{0x2329, 0x0000},
	{0x23B4, 0x0000},
	{0x2768, 0x0000},
	{0x276A, 0x0000},
	{0x276C, 0x0000},
	{0x276E, 0x0000},
	{0x2770, 0x0000},
	{0x2772, 0x0000},
	{0x2774, 0x0000},
	{0x27E6, 0x0000},
	{0x27E8, 0x0000},
	{0x27EA, 0x0000},
	{0x2983, 0x0000},
	{0x2985, 0x0000},
	{0x2987, 0x0000},
	{0x2989, 0x0000},
	{0x298B, 0x0000},
	{0x298D, 0x0000},
	{0x298F, 0x0000},
	{0x2991, 0x0000},
	{0x2993, 0x0000},
	{0x2995, 0x0000},
	{0x2997, 0x0000},
	{0x29D8, 0x0000},
	{0x29DA, 0x0000},
	{0x29FC, 0x0000},
	{0x3008, 0x0000},
	{0x300A, 0x0000},
	{0x300C, 0x0000},
	{0x300E, 0x0000},
	{0x3010, 0x0000},
	{0x3014, 0x0000},
	{0x3016, 0x0000},
	{0x3018, 0x0000},
	{0x301A, 0x0000},
	{0x301D, 0x0000},
	{0xFD3E, 0x0000},
	{0xFE35, 0x0000},
	{0xFE37, 0x0000},
	{0xFE39, 0x0000},
	{0xFE3B, 0x0000},
	{0xFE3D, 0x0000},
	{0xFE3F, 0x0000},
	{0xFE41, 0x0000},
	{0xFE43, 0x0000},
	{0xFE47, 0x0000},
	{0xFE59, 0x0000},
	{0xFE5B, 0x0000},
	{0xFE5D, 0x0000},
	{0xFF08, 0x0000},
	{0xFF3B, 0x0000},
	{0xFF5B, 0x0000},
	{0xFF5F, 0x0000},
	{0xFF62, 0x0000}
};

CATEGORY_CHAR_RANGE PeRanges[] =
{
	{0x0029, 0x0000},
	{0x005D, 0x0000},
	{0x007D, 0x0000},
	{0x0F3B, 0x0000},
	{0x0F3D, 0x0000},
	{0x169C, 0x0000},
	{0x2046, 0x0000},
	{0x207E, 0x0000},
	{0x208E, 0x0000},
	{0x232A, 0x0000},
	{0x23B5, 0x0000},
	{0x2769, 0x0000},
	{0x276B, 0x0000},
	{0x276D, 0x0000},
	{0x276F, 0x0000},
	{0x2771, 0x0000},
	{0x2773, 0x0000},
	{0x2775, 0x0000},
	{0x27E7, 0x0000},
	{0x27E9, 0x0000},
	{0x27EB, 0x0000},
	{0x2984, 0x0000},
	{0x2986, 0x0000},
	{0x2988, 0x0000},
	{0x298A, 0x0000},
	{0x298C, 0x0000},
	{0x298E, 0x0000},
	{0x2990, 0x0000},
	{0x2992, 0x0000},
	{0x2994, 0x0000},
	{0x2996, 0x0000},
	{0x2998, 0x0000},
	{0x29D9, 0x0000},
	{0x29DB, 0x0000},
	{0x29FD, 0x0000},
	{0x3009, 0x0000},
	{0x300B, 0x0000},
	{0x300D, 0x0000},
	{0x300F, 0x0000},
	{0x3011, 0x0000},
	{0x3015, 0x0000},
	{0x3017, 0x0000},
	{0x3019, 0x0000},
	{0x301B, 0x0000},
	{0x301E, 0x0000},
	{0x301F, 0x0000},
	{0xFD3F, 0x0000},
	{0xFE36, 0x0000},
	{0xFE38, 0x0000},
	{0xFE3A, 0x0000},
	{0xFE3C, 0x0000},
	{0xFE3E, 0x0000},
	{0xFE40, 0x0000},
	{0xFE42, 0x0000},
	{0xFE44, 0x0000},
	{0xFE48, 0x0000},
	{0xFE5A, 0x0000},
	{0xFE5C, 0x0000},
	{0xFE5E, 0x0000},
	{0xFF09, 0x0000},
	{0xFF3D, 0x0000},
	{0xFF5D, 0x0000},
	{0xFF60, 0x0000},
	{0xFF63, 0x0000}
};

CATEGORY_CHAR_RANGE PiRanges[] =
{
	{0x00AB, 0x0000},
	{0x2018, 0x0000},
	{0x201B, 0x201C},
	{0x201F, 0x0000},
	{0x2039, 0x0000}
};

CATEGORY_CHAR_RANGE PfRanges[] =
{
	{0x00BB, 0x0000},
	{0x2019, 0x0000},
	{0x201D, 0x0000},
	{0x203A, 0x0000}
};

CATEGORY_CHAR_RANGE PoRanges[] =
{
	{0x0021, 0x0023},
	{0x0025, 0x0027},
	{0x002A, 0x0000},
	{0x002C, 0x0000},
	{0x002E, 0x002F},
	{0x003A, 0x003B},
	{0x003F, 0x0040},
	{0x005C, 0x0000},
	{0x00A1, 0x0000},
	{0x00B7, 0x0000},
	{0x00BF, 0x0000},
	{0x037E, 0x0000},
	{0x0387, 0x0000},
	{0x055A, 0x055F},
	{0x0589, 0x0000},
	{0x05BE, 0x0000},
	{0x05C0, 0x0000},
	{0x05C3, 0x0000},
	{0x05F3, 0x05F4},
	{0x060C, 0x060D},
	{0x061B, 0x0000},
	{0x061F, 0x0000},
	{0x066A, 0x066D},
	{0x06D4, 0x0000},
	{0x0700, 0x070D},
	{0x0964, 0x0965},
	{0x0970, 0x0000},
	{0x0DF4, 0x0000},
	{0x0E4F, 0x0000},
	{0x0E5A, 0x0E5B},
	{0x0F04, 0x0F12},
	{0x0F85, 0x0000},
	{0x104A, 0x104F},
	{0x10FB, 0x0000},
	{0x1361, 0x1368},
	{0x166D, 0x166E},
	{0x16EB, 0x16ED},
	{0x1735, 0x1736},
	{0x17D4, 0x17D6},
	{0x17D8, 0x17DA},
	{0x1800, 0x1805},
	{0x1807, 0x180A},
	{0x1944, 0x1945},
	{0x2016, 0x2017},
	{0x2020, 0x2027},
	{0x2030, 0x2038},
	{0x203B, 0x203E},
	{0x2041, 0x2043},
	{0x2047, 0x2051},
	{0x2053, 0x0000},
	{0x2057, 0x0000},
	{0x23B6, 0x0000},
	{0x3001, 0x3003},
	{0x303D, 0x0000},
	{0xFE30, 0x0000},
	{0xFE45, 0xFE46},
	{0xFE49, 0xFE4C},
	{0xFE50, 0xFE52},
	{0xFE54, 0xFE57},
	{0xFE5F, 0xFE61},
	{0xFE68, 0x0000},
	{0xFE6A, 0xFE6B},
	{0xFF01, 0xFF03},
	{0xFF05, 0xFF07},
	{0xFF0A, 0x0000},
	{0xFF0C, 0x0000},
	{0xFF0E, 0xFF0F},
	{0xFF1A, 0xFF1B},
	{0xFF1F, 0xFF20},
	{0xFF3C, 0x0000},
	{0xFF61, 0x0000},
	{0xFF64, 0x0000}
};

CATEGORY_CHAR_RANGE ZsRanges[] =
{
	{0x0020, 0x0000},
	{0x00A0, 0x0000},
	{0x1680, 0x0000},
	{0x180E, 0x0000},
	{0x2000, 0x200B},
	{0x202F, 0x0000},
	{0x205F, 0x0000},
	{0x3000, 0x0000}
};

CATEGORY_CHAR_RANGE ZlRanges[] =
{
	{0x2028, 0x0000}
};

CATEGORY_CHAR_RANGE ZpRanges[] =
{
	{0x2029, 0x0000}
};

CATEGORY_CHAR_RANGE SmRanges[] =
{
	{0x002B, 0x0000},
	{0x003C, 0x003E},
	{0x007C, 0x0000},
	{0x007E, 0x0000},
	{0x00AC, 0x0000},
	{0x00B1, 0x0000},
	{0x00D7, 0x0000},
	{0x00F7, 0x0000},
	{0x03F6, 0x0000},
	{0x2044, 0x0000},
	{0x2052, 0x0000},
	{0x207A, 0x207C},
	{0x208A, 0x208C},
	{0x2140, 0x2144},
	{0x214B, 0x0000},
	{0x2190, 0x2194},
	{0x219A, 0x219B},
	{0x21A0, 0x0000},
	{0x21A3, 0x0000},
	{0x21A6, 0x0000},
	{0x21AE, 0x0000},
	{0x21CE, 0x21CF},
	{0x21D2, 0x0000},
	{0x21D4, 0x0000},
	{0x21F4, 0x21FF},
	{0x2200, 0x22FF},
	{0x2308, 0x230B},
	{0x2320, 0x2321},
	{0x237C, 0x0000},
	{0x239B, 0x23B3},
	{0x25B7, 0x0000},
	{0x25C1, 0x0000},
	{0x25F8, 0x25FF},
	{0x266F, 0x0000},
	{0x27D0, 0x27E5},
	{0x27F0, 0x27FF},
	{0x2900, 0x2982},
	{0x2999, 0x29D7},
	{0x29DC, 0x29FB},
	{0x29FE, 0x2AFF},
	{0xFB29, 0x0000},
	{0xFE62, 0x0000},
	{0xFE64, 0xFE66},
	{0xFF0B, 0x0000},
	{0xFF1C, 0xFF1E},
	{0xFF5C, 0x0000},
	{0xFF5E, 0x0000},
	{0xFFE2, 0x0000},
	{0xFFE9, 0xFFEC}
};

CATEGORY_CHAR_RANGE ScRanges[] =
{
	{0x0024, 0x0000},
	{0x00A2, 0x00A5},
	{0x09F2, 0x09F3},
	{0x0AF1, 0x0000},
	{0x0BF9, 0x0000},
	{0x0E3F, 0x0000},
	{0x17DB, 0x0000},
	{0x20A0, 0x20B1},
	{0xFDFC, 0x0000},
	{0xFE69, 0x0000},
	{0xFF04, 0x0000},
	{0xFFE0, 0xFFE1},
	{0xFFE5, 0xFFE6}
};

CATEGORY_CHAR_RANGE SkRanges[] =
{
	{0x005E, 0x0000},
	{0x0060, 0x0000},
	{0x00A8, 0x0000},
	{0x00AF, 0x0000},
	{0x00B4, 0x0000},
	{0x00B8, 0x0000},
	{0x02C2, 0x02C5},
	{0x02D2, 0x02DF},
	{0x02E5, 0x02FF},
	{0x0374, 0x0375},
	{0x0384, 0x0385},
	{0x1FBD, 0x0000},
	{0x1FBF, 0x1FC1},
	{0x1FCD, 0x1FCF},
	{0x1FDD, 0x1FDF},
	{0x1FED, 0x1FEF},
	{0x1FFD, 0x0000},
	{0x1FFE, 0x0000},
	{0x309B, 0x309C},
	{0xFF3E, 0x0000},
	{0xFF40, 0x0000},
	{0xFFE3, 0x0000}
};

CATEGORY_CHAR_RANGE SoRanges[] =
{
	{0x00A6, 0x00A7},
	{0x00A9, 0x0000},
	{0x00AE, 0x0000},
	{0x00B0, 0x0000},
	{0x00B6, 0x0000},
	{0x0482, 0x0000},
	{0x060E, 0x060F},
	{0x06E9, 0x0000},
	{0x06FD, 0x06FE},
	{0x09FA, 0x0000},
	{0x0B70, 0x0000},
	{0x0BF3, 0x0BFA},
	{0x0F01, 0x0F03},
	{0x0F13, 0x0F17},
	{0x0F1A, 0x0F1F},
	{0x0F34, 0x0000},
	{0x0F36, 0x0000},
	{0x0F38, 0x0000},
	{0x0FBE, 0x0FBF},
	{0x0FC0, 0x0FC5},
	{0x0FC7, 0x0FCC},
	{0x0FCF, 0x0000},
	{0x1940, 0x0000},
	{0x19E0, 0x19FF},
	{0x2100, 0x2101},
	{0x2103, 0x2106},
	{0x2108, 0x2109},
	{0x2114, 0x0000},
	{0x2116, 0x2118},
	{0x211E, 0x211F},
	{0x2120, 0x2123},
	{0x2125, 0x0000},
	{0x2127, 0x0000},
	{0x2129, 0x0000},
	{0x212E, 0x0000},
	{0x2132, 0x0000},
	{0x213A, 0x213B},
	{0x214A, 0x0000},
	{0x2195, 0x2199},
	{0x219C, 0x219F},
	{0x21A1, 0x21A2},
	{0x21A4, 0x21A5},
	{0x21A7, 0x21AD},
	{0x21AF, 0x21CD},
	{0x21D0, 0x21D1},
	{0x21D3, 0x0000},
	{0x21D5, 0x21F3},
	{0x2300, 0x2307},
	{0x230C, 0x231F},
	{0x2322, 0x2328},
	{0x232B, 0x237B},
	{0x237D, 0x239A},
	{0x23B7, 0x23D0},
	{0x2400, 0x2426},
	{0x2440, 0x244A},
	{0x249C, 0x24E9},
	{0x2500, 0x25C0},
	{0x25C2, 0x25F7},
	{0x2600, 0x2617},
	{0x2619, 0x266E},
	{0x2670, 0x267D},
	{0x2680, 0x2691},
	{0x26A0, 0x26A1},
	{0x2701, 0x2704},
	{0x2706, 0x2709},
	{0x270C, 0x2727},
	{0x2729, 0x274B},
	{0x274D, 0x0000},
	{0x274F, 0x2752},
	{0x2756, 0x0000},
	{0x2758, 0x275E},
	{0x2761, 0x2767},
	{0x2794, 0x0000},
	{0x2798, 0x27AF},
	{0x27B1, 0x27BE},
	{0x2800, 0x28FF},
	{0x2B00, 0x2B0D},
	{0x2E80, 0x2E99},
	{0x2E9B, 0x2EF3},
	{0x2F00, 0x2FD5},
	{0x2FF0, 0x2FFB},
	{0x3004, 0x0000},
	{0x3012, 0x3013},
	{0x3020, 0x0000},
	{0x3036, 0x3037},
	{0x303E, 0x303F},
	{0x3190, 0x3191},
	{0x3196, 0x319F},
	{0x3200, 0x321E},
	{0x322A, 0x3243},
	{0x3250, 0x0000},
	{0x3260, 0x327D},
	{0x327F, 0x0000},
	{0x328A, 0x32FE},
	{0x3300, 0x33FF},
	{0x4DC0, 0x4DFF},
	{0xA490, 0xA4C6},
	{0xFDFD, 0x0000},
	{0xFFE4, 0x0000},
	{0xFFE8, 0x0000},
	{0xFFED, 0xFFEE},
	{0xFFFC, 0xFFFD}
};

CATEGORY_CHAR_RANGE CcRanges[] =
{
	{0x0001, 0x001F},
	{0x007F, 0x009F}
};

CATEGORY_CHAR_RANGE CfRanges[] =
{
	{0x00AD, 0x0000},
	{0x0600, 0x0603},
	{0x06DD, 0x0000},
	{0x070F, 0x0000},
	{0x17B4, 0x17B5},
	{0x200C, 0x200F},
	{0x202A, 0x202E},
	{0x2060, 0x2063},
	{0x206A, 0x206F},
	{0xFEFF, 0x0000},
	{0xFFF9, 0xFFFB}
};

CATEGORY_CHAR_RANGE CoRanges[] =
{
	{0xE000, 0x0000},
	{0xF8FF, 0x0000}
};

CATEGORY_CHAR_RANGE *	CnRanges = NULL;

typedef struct CATEGORY_CHARS
{
	FLMUINT						uiNumArrays;
	FLMUINT *					puiRangeArraySizes;
	CATEGORY_CHAR_RANGE **	ppRangeArrays;
} CATEGORY_CHARS;

#define	LuRangeSize	(sizeof( LuRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	LlRangeSize	(sizeof( LlRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	LtRangeSize	(sizeof( LtRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	LmRangeSize	(sizeof( LmRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	LoRangeSize	(sizeof( LoRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT LCategoryRangeSizes [] =
		{LuRangeSize, LlRangeSize, LtRangeSize, LmRangeSize, LoRangeSize};

CATEGORY_CHAR_RANGE * LCategoryRanges [] = 
	{LuRanges, LlRanges, LtRanges, LmRanges, LoRanges};
	
CATEGORY_CHARS LCategory = { 5, &LCategoryRangeSizes [0], &LCategoryRanges [0]};
CATEGORY_CHARS LuCategory = { 1, &LCategoryRangeSizes [0], &LCategoryRanges [0]};
CATEGORY_CHARS LlCategory = { 1, &LCategoryRangeSizes [1], &LCategoryRanges [1]};
CATEGORY_CHARS LtCategory = { 1, &LCategoryRangeSizes [2], &LCategoryRanges [2]};
CATEGORY_CHARS LmCategory = { 1, &LCategoryRangeSizes [3], &LCategoryRanges [3]};
CATEGORY_CHARS LoCategory = { 1, &LCategoryRangeSizes [4], &LCategoryRanges [4]};

#define	MnRangeSize	(sizeof( MnRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	McRangeSize	(sizeof( McRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	MeRangeSize	(sizeof( MeRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT MCategoryRangeSizes [] = {MnRangeSize, McRangeSize, MeRangeSize};

CATEGORY_CHAR_RANGE * MCategoryRanges[] = {MnRanges, McRanges, MeRanges};

CATEGORY_CHARS MCategory = { 3, &MCategoryRangeSizes [0], &MCategoryRanges [0]};
CATEGORY_CHARS MnCategory = { 1, &MCategoryRangeSizes [0], &MCategoryRanges [0]};
CATEGORY_CHARS McCategory = { 1, &MCategoryRangeSizes [1], &MCategoryRanges [1]};
CATEGORY_CHARS MeCategory = { 1, &MCategoryRangeSizes [2], &MCategoryRanges [2]};

#define	DigitRangeSize	(sizeof( DigitRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	Nd2RangeSize	(sizeof( Nd2Ranges) / sizeof( CATEGORY_CHAR_RANGE))
#define	NlRangeSize		(sizeof( NlRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	NoRangeSize		(sizeof( NoRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT NCategoryRangeSizes [] =
	{DigitRangeSize, Nd2RangeSize, NlRangeSize, NoRangeSize};

CATEGORY_CHAR_RANGE * NCategoryRanges[] =
	{DigitRanges, Nd2Ranges, NlRanges, NoRanges};

CATEGORY_CHARS NCategory = { 4, &NCategoryRangeSizes [0], &NCategoryRanges [0]};
CATEGORY_CHARS DigitCategory = { 1, &NCategoryRangeSizes [0], &NCategoryRanges [0]};
CATEGORY_CHARS NdCategory = { 2, &NCategoryRangeSizes [0], &NCategoryRanges [0]};
CATEGORY_CHARS Nd2Category = { 1, &NCategoryRangeSizes [1], &NCategoryRanges [1]};
CATEGORY_CHARS NlCategory = { 1, &NCategoryRangeSizes [2], &NCategoryRanges [2]};
CATEGORY_CHARS NoCategory = { 1, &NCategoryRangeSizes [3], &NCategoryRanges [3]};

#define	PcRangeSize	(sizeof( PcRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	PdRangeSize	(sizeof( PdRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	PsRangeSize	(sizeof( PsRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	PeRangeSize	(sizeof( PeRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	PiRangeSize	(sizeof( PiRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	PfRangeSize	(sizeof( PfRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	PoRangeSize	(sizeof( PoRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT PCategoryRangeSizes [] =
	{PcRangeSize, PdRangeSize, PsRangeSize, PeRangeSize,
	 PiRangeSize, PfRangeSize, PoRangeSize};

CATEGORY_CHAR_RANGE * PCategoryRanges[] =
	{PcRanges, PdRanges, PsRanges, PeRanges, PiRanges, PfRanges, PoRanges};

CATEGORY_CHARS PCategory = { 7, &PCategoryRangeSizes [0], &PCategoryRanges [0]};
CATEGORY_CHARS PcCategory = { 1, &PCategoryRangeSizes [0], &PCategoryRanges [0]};
CATEGORY_CHARS PdCategory = { 1, &PCategoryRangeSizes [1], &PCategoryRanges [1]};
CATEGORY_CHARS PsCategory = { 1, &PCategoryRangeSizes [2], &PCategoryRanges [2]};
CATEGORY_CHARS PeCategory = { 1, &PCategoryRangeSizes [3], &PCategoryRanges [3]};
CATEGORY_CHARS PiCategory = { 1, &PCategoryRangeSizes [4], &PCategoryRanges [4]};
CATEGORY_CHARS PfCategory = { 1, &PCategoryRangeSizes [5], &PCategoryRanges [5]};
CATEGORY_CHARS PoCategory = { 1, &PCategoryRangeSizes [6], &PCategoryRanges [6]};

#define	ZsRangeSize	(sizeof( ZsRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	ZlRangeSize	(sizeof( ZlRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	ZpRangeSize	(sizeof( ZpRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT ZCategoryRangeSizes [] = {ZsRangeSize, ZlRangeSize, ZpRangeSize};

CATEGORY_CHAR_RANGE * ZCategoryRanges[] = {ZsRanges, ZlRanges, ZpRanges};

CATEGORY_CHARS ZCategory = { 3, &ZCategoryRangeSizes [0], &ZCategoryRanges [0]};
CATEGORY_CHARS ZsCategory = { 1, &ZCategoryRangeSizes [0], &ZCategoryRanges [0]};
CATEGORY_CHARS ZlCategory = { 1, &ZCategoryRangeSizes [1], &ZCategoryRanges [1]};
CATEGORY_CHARS ZpCategory = { 1, &ZCategoryRangeSizes [2], &ZCategoryRanges [2]};

#define	SmRangeSize	(sizeof( SmRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	ScRangeSize	(sizeof( ScRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	SkRangeSize	(sizeof( SkRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	SoRangeSize	(sizeof( SoRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT SCategoryRangeSizes [] = {SmRangeSize, ScRangeSize, SkRangeSize, SoRangeSize};

CATEGORY_CHAR_RANGE * SCategoryRanges[] = {SmRanges, ScRanges, SkRanges, SoRanges};

CATEGORY_CHARS SCategory = { 4, &SCategoryRangeSizes [0], &SCategoryRanges [0]};
CATEGORY_CHARS SmCategory = { 1, &SCategoryRangeSizes [0], &SCategoryRanges [0]};
CATEGORY_CHARS ScCategory = { 1, &SCategoryRangeSizes [1], &SCategoryRanges [1]};
CATEGORY_CHARS SkCategory = { 1, &SCategoryRangeSizes [2], &SCategoryRanges [2]};
CATEGORY_CHARS SoCategory = { 1, &SCategoryRangeSizes [3], &SCategoryRanges [3]};

#define	CcRangeSize	(sizeof( CcRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	CfRangeSize	(sizeof( CfRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	CoRangeSize	(sizeof( CoRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	CnRangeSize	0

FLMUINT CCategoryRangeSizes [] = {CcRangeSize, CfRangeSize, CoRangeSize, CnRangeSize};

CATEGORY_CHAR_RANGE * CCategoryRanges[] = {CcRanges, CfRanges, CoRanges, CnRanges};

CATEGORY_CHARS CCategory = { 4, &CCategoryRangeSizes [0], &CCategoryRanges [0]};
CATEGORY_CHARS CcCategory = { 1, &CCategoryRangeSizes [0], &CCategoryRanges [0]};
CATEGORY_CHARS CfCategory = { 1, &CCategoryRangeSizes [1], &CCategoryRanges [1]};
CATEGORY_CHARS CoCategory = { 1, &CCategoryRangeSizes [2], &CCategoryRanges [2]};
CATEGORY_CHARS CnCategory = { 1, &CCategoryRangeSizes [3], &CCategoryRanges [3]};

CATEGORY_CHAR_RANGE LetterRanges [] =
{
	{0x0041, 0x005A},
	{0x0061, 0x007A},
	{0x00C0, 0x00D6},
	{0x00D8, 0x00F6},
	{0x00F8, 0x00FF},
	{0x0100, 0x0131},
	{0x0134, 0x013E},
	{0x0141, 0x0148},
	{0x014A, 0x017E},
	{0x0180, 0x01C3},
	{0x01CD, 0x01F0},
	{0x01F4, 0x01F5},
	{0x01FA, 0x0217},
	{0x0250, 0x02A8},
	{0x02BB, 0x02C1},
	{0x0386, 0x0000},
	{0x0388, 0x038A},
	{0x038C, 0x0000},
	{0x038E, 0x03A1},
	{0x03A3, 0x03CE},
	{0x03D0, 0x03D6},
	{0x03DA, 0x0000},
	{0x03DC, 0x0000},
	{0x03DE, 0x0000},
	{0x03E0, 0x0000},
	{0x03E2, 0x03F3},
	{0x0401, 0x040C},
	{0x040E, 0x044F},
	{0x0451, 0x045C},
	{0x045E, 0x0481},
	{0x0490, 0x04C4},
	{0x04C7, 0x04C8},
	{0x04CB, 0x04CC},
	{0x04D0, 0x04EB},
	{0x04EE, 0x04F5},
	{0x04F8, 0x04F9},
	{0x0531, 0x0556},
	{0x0559, 0x0000},
	{0x0561, 0x0586},
	{0x05D0, 0x05EA},
	{0x05F0, 0x05F2},
	{0x0621, 0x063A},
	{0x0641, 0x064A},
	{0x0671, 0x06B7},
	{0x06BA, 0x06BE},
	{0x06C0, 0x06CE},
	{0x06D0, 0x06D3},
	{0x06D5, 0x0000},
	{0x06E5, 0x06E6},
	{0x0905, 0x0939},
	{0x093D, 0x0000},
	{0x0958, 0x0961},
	{0x0985, 0x098C},
	{0x098F, 0x0990},
	{0x0993, 0x09A8},
	{0x09AA, 0x09B0},
	{0x09B2, 0x0000},
	{0x09B6, 0x09B9},
	{0x09DC, 0x09DD},
	{0x09DF, 0x09E1},
	{0x09F0, 0x09F1},
	{0x0A05, 0x0A0A},
	{0x0A0F, 0x0A10},
	{0x0A13, 0x0A28},
	{0x0A2A, 0x0A30},
	{0x0A32, 0x0A33},
	{0x0A35, 0x0A36},
	{0x0A38, 0x0A39},
	{0x0A59, 0x0A5C},
	{0x0A5E, 0x0000},
	{0x0A72, 0x0A74},
	{0x0A85, 0x0A8B},
	{0x0A8D, 0x0000},
	{0x0A8F, 0x0A91},
	{0x0A93, 0x0AA8},
	{0x0AAA, 0x0AB0},
	{0x0AB2, 0x0AB3},
	{0x0AB5, 0x0AB9},
	{0x0ABD, 0x0000},
	{0x0AE0, 0x0000},
	{0x0B05, 0x0B0C},
	{0x0B0F, 0x0B10},
	{0x0B13, 0x0B28},
	{0x0B2A, 0x0B30},
	{0x0B32, 0x0B33},
	{0x0B36, 0x0B39},
	{0x0B3D, 0x0000},
	{0x0B5C, 0x0B5D},
	{0x0B5F, 0x0B61},
	{0x0B85, 0x0B8A},
	{0x0B8E, 0x0B90},
	{0x0B92, 0x0B95},
	{0x0B99, 0x0B9A},
	{0x0B9C, 0x0000},
	{0x0B9E, 0x0B9F},
	{0x0BA3, 0x0BA4},
	{0x0BA8, 0x0BAA},
	{0x0BAE, 0x0BB5},
	{0x0BB7, 0x0BB9},
	{0x0C05, 0x0C0C},
	{0x0C0E, 0x0C10},
	{0x0C12, 0x0C28},
	{0x0C2A, 0x0C33},
	{0x0C35, 0x0C39},
	{0x0C60, 0x0C61},
	{0x0C85, 0x0C8C},
	{0x0C8E, 0x0C90},
	{0x0C92, 0x0CA8},
	{0x0CAA, 0x0CB3},
	{0x0CB5, 0x0CB9},
	{0x0CDE, 0x0000},
	{0x0CE0, 0x0CE1},
	{0x0D05, 0x0D0C},
	{0x0D0E, 0x0D10},
	{0x0D12, 0x0D28},
	{0x0D2A, 0x0D39},
	{0x0D60, 0x0D61},
	{0x0E01, 0x0E2E},
	{0x0E30, 0x0000},
	{0x0E32, 0x0E33},
	{0x0E40, 0x0E45},
	{0x0E81, 0x0E82},
	{0x0E84, 0x0000},
	{0x0E87, 0x0E88},
	{0x0E8A, 0x0000},
	{0x0E8D, 0x0000},
	{0x0E94, 0x0E97},
	{0x0E99, 0x0E9F},
	{0x0EA1, 0x0EA3},
	{0x0EA5, 0x0000},
	{0x0EA7, 0x0000},
	{0x0EAA, 0x0EAB},
	{0x0EAD, 0x0EAE},
	{0x0EB0, 0x0000},
	{0x0EB2, 0x0EB3},
	{0x0EBD, 0x0000},
	{0x0EC0, 0x0EC4},
	{0x0F40, 0x0F47},
	{0x0F49, 0x0F69},
	{0x10A0, 0x10C5},
	{0x10D0, 0x10F6},
	{0x1100, 0x0000},
	{0x1102, 0x1103},
	{0x1105, 0x1107},
	{0x1109, 0x0000},
	{0x110B, 0x110C},
	{0x110E, 0x1112},
	{0x113C, 0x0000},
	{0x113E, 0x0000},
	{0x1140, 0x0000},
	{0x114C, 0x0000},
	{0x114E, 0x0000},
	{0x1150, 0x0000},
	{0x1154, 0x1155},
	{0x1159, 0x0000},
	{0x115F, 0x1161},
	{0x1163, 0x0000},
	{0x1165, 0x0000},
	{0x1167, 0x0000},
	{0x1169, 0x0000},
	{0x116D, 0x116E},
	{0x1172, 0x1173},
	{0x1175, 0x0000},
	{0x119E, 0x0000},
	{0x11A8, 0x0000},
	{0x11AB, 0x0000},
	{0x11AE, 0x11AF},
	{0x11B7, 0x11B8},
	{0x11BA, 0x0000},
	{0x11BC, 0x11C2},
	{0x11EB, 0x0000},
	{0x11F0, 0x0000},
	{0x11F9, 0x0000},
	{0x1E00, 0x1E9B},
	{0x1EA0, 0x1EF9},
	{0x1F00, 0x1F15},
	{0x1F18, 0x1F1D},
	{0x1F20, 0x1F45},
	{0x1F48, 0x1F4D},
	{0x1F50, 0x1F57},
	{0x1F59, 0x0000},
	{0x1F5B, 0x0000},
	{0x1F5D, 0x0000},
	{0x1F5F, 0x1F7D},
	{0x1F80, 0x1FB4},
	{0x1FB6, 0x1FBC},
	{0x1FBE, 0x0000},
	{0x1FC2, 0x1FC4},
	{0x1FC6, 0x1FCC},
	{0x1FD0, 0x1FD3},
	{0x1FD6, 0x1FDB},
	{0x1FE0, 0x1FEC},
	{0x1FF2, 0x1FF4},
	{0x1FF6, 0x1FFC},
	{0x2126, 0x0000},
	{0x212A, 0x212B},
	{0x212E, 0x0000},
	{0x2180, 0x2182},
	{0x3007, 0x0000},
	{0x3021, 0x3029},
	{0x3041, 0x3094},
	{0x30A1, 0x30FA},
	{0x3105, 0x312C},
	{0x4E00, 0x9FA5},
	{0xAC00, 0xD7A3}
};

CATEGORY_CHAR_RANGE CombiningRanges [] =
{
	{0x0300, 0x0345},
	{0x0360, 0x0361},
	{0x0483, 0x0486},
	{0x0591, 0x05A1},
	{0x05A3, 0x05B9},
	{0x05BB, 0x05BD},
	{0x05BF, 0x0000},
	{0x05C1, 0x05C2},
	{0x05C4, 0x0000},
	{0x064B, 0x0652},
	{0x0670, 0x0000},
	{0x06D6, 0x06DC},
	{0x06DD, 0x06DF},
	{0x06E0, 0x06E4},
	{0x06E7, 0x06E8},
	{0x06EA, 0x06ED},
	{0x0901, 0x0903},
	{0x093C, 0x0000},
	{0x093E, 0x094C},
	{0x094D, 0x0000},
	{0x0951, 0x0954},
	{0x0962, 0x0963},
	{0x0981, 0x0983},
	{0x09BC, 0x0000},
	{0x09BE, 0x0000},
	{0x09BF, 0x0000},
	{0x09C0, 0x09C4},
	{0x09C7, 0x09C8},
	{0x09CB, 0x09CD},
	{0x09D7, 0x0000},
	{0x09E2, 0x09E3},
	{0x0A02, 0x0000},
	{0x0A3C, 0x0000},
	{0x0A3E, 0x0000},
	{0x0A3F, 0x0000},
	{0x0A40, 0x0A42},
	{0x0A47, 0x0A48},
	{0x0A4B, 0x0A4D},
	{0x0A70, 0x0A71},
	{0x0A81, 0x0A83},
	{0x0ABC, 0x0000},
	{0x0ABE, 0x0AC5},
	{0x0AC7, 0x0AC9},
	{0x0ACB, 0x0ACD},
	{0x0B01, 0x0B03},
	{0x0B3C, 0x0000},
	{0x0B3E, 0x0B43},
	{0x0B47, 0x0B48},
	{0x0B4B, 0x0B4D},
	{0x0B56, 0x0B57},
	{0x0B82, 0x0B83},
	{0x0BBE, 0x0BC2},
	{0x0BC6, 0x0BC8},
	{0x0BCA, 0x0BCD},
	{0x0BD7, 0x0000},
	{0x0C01, 0x0C03},
	{0x0C3E, 0x0C44},
	{0x0C46, 0x0C48},
	{0x0C4A, 0x0C4D},
	{0x0C55, 0x0C56},
	{0x0C82, 0x0C83},
	{0x0CBE, 0x0CC4},
	{0x0CC6, 0x0CC8},
	{0x0CCA, 0x0CCD},
	{0x0CD5, 0x0CD6},
	{0x0D02, 0x0D03},
	{0x0D3E, 0x0D43},
	{0x0D46, 0x0D48},
	{0x0D4A, 0x0D4D},
	{0x0D57, 0x0000},
	{0x0E31, 0x0000},
	{0x0E34, 0x0E3A},
	{0x0E47, 0x0E4E},
	{0x0EB1, 0x0000},
	{0x0EB4, 0x0EB9},
	{0x0EBB, 0x0EBC},
	{0x0EC8, 0x0ECD},
	{0x0F18, 0x0F19},
	{0x0F35, 0x0000},
	{0x0F37, 0x0000},
	{0x0F39, 0x0000},
	{0x0F3E, 0x0000},
	{0x0F3F, 0x0000},
	{0x0F71, 0x0F84},
	{0x0F86, 0x0F8B},
	{0x0F90, 0x0F95},
	{0x0F97, 0x0000},
	{0x0F99, 0x0FAD},
	{0x0FB1, 0x0FB7},
	{0x0FB9, 0x0000},
	{0x20D0, 0x20DC},
	{0x20E1, 0x0000},
	{0x302A, 0x302F},
	{0x3099, 0x0000},
	{0x309A, 0x0000}
};

CATEGORY_CHAR_RANGE ExtenderRanges [] =
{
	{0x00B7, 0x0000},
	{0x02D0, 0x0000},
	{0x02D1, 0x0000},
	{0x0387, 0x0000},
	{0x0640, 0x0000},
	{0x0E46, 0x0000},
	{0x0EC6, 0x0000},
	{0x3005, 0x0000},
	{0x3031, 0x3035},
	{0x309D, 0x309E},
	{0x30FC, 0x30FE}
};

#define	LetterRangeSize		(sizeof( LetterRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	CombiningRangeSize	(sizeof( CombiningRanges) / sizeof( CATEGORY_CHAR_RANGE))
#define	ExtenderRangeSize		(sizeof( ExtenderRanges) / sizeof( CATEGORY_CHAR_RANGE))

FLMUINT NameCharCategoryRangeSizes [] = 
	{LetterRangeSize, CombiningRangeSize, ExtenderRangeSize, DigitRangeSize};

CATEGORY_CHAR_RANGE * NameCharCategoryRanges[] =
	{LetterRanges, CombiningRanges, ExtenderRanges, DigitRanges};

CATEGORY_CHARS NameCharCategory = { 4, &NameCharCategoryRangeSizes [0], &NameCharCategoryRanges [0]};
CATEGORY_CHARS LetterCategory = { 1, &NameCharCategoryRangeSizes [0], &NameCharCategoryRanges [0]};
CATEGORY_CHARS CombiningCategory = { 1, &NameCharCategoryRangeSizes [1], &NameCharCategoryRanges [1]};
CATEGORY_CHARS ExtenderCategory = { 1, &NameCharCategoryRangeSizes [2], &NameCharCategoryRanges [2]};

typedef struct REGEXP_LITERAL
{
	FLMUNICODE *	puzLiteral;
	FLMUINT			uiNumChars;
} REGEXP_LITERAL;

typedef struct CHAR_RANGE
{
	FLMBOOL			bNegatedRange;
	FLMUNICODE		uzLowChar;
	FLMUNICODE		uzHighChar;
	CHAR_RANGE *	pNext;
	CHAR_RANGE *	pPrev;
} CHAR_RANGE;

typedef struct CHAR_LIST
{
	FLMBOOL			bNegatedChars;
	FLMUNICODE *	puzChars;
	FLMUINT			uiNumChars;
	CHAR_LIST *		pNext;
	CHAR_LIST *		pPrev;
} CHAR_LIST;

typedef struct CHAR_CATEGORY
{
	FLMBOOL				bNegatedCategory;
	CATEGORY_CHARS *	pCategoryChars;
	CHAR_CATEGORY *	pNext;
	CHAR_CATEGORY *	pPrev;
} CHAR_CATEGORY;

typedef struct REG_EXP *	REG_EXP_p;

typedef struct CHAR_CLASS
{
	FLMBOOL				bNegatedClass;
	CHAR_LIST *			pFirstCharList;
	CHAR_LIST *			pLastCharList;
	CHAR_RANGE *		pFirstCharRange;
	CHAR_RANGE *		pLastCharRange;
	CHAR_CATEGORY *	pFirstCharCategory;
	CHAR_CATEGORY *	pLastCharCategory;
	CHAR_CLASS *		pFirstCharClass;
	CHAR_CLASS *		pLastCharClass;
	CHAR_CLASS *		pSubtractionClass;
	CHAR_CLASS *		pNext;
	CHAR_CLASS *		pPrev;
} CHAR_CLASS;

typedef struct REG_EXP_BRANCH
{
	REG_EXP_p			pParentExpr;
	REG_EXP_p			pFirstExpr;
	REG_EXP_p			pLastExpr;
	REG_EXP_BRANCH *	pNextBranch;
	REG_EXP_BRANCH *	pPrevBranch;
} REG_EXP_BRANCH;

typedef struct REG_EXP_ALTERNATIVE
{
	REG_EXP_BRANCH *	pFirstBranch;
	REG_EXP_BRANCH *	pLastBranch;
} REG_EXP_ALTERNATIVE;

typedef struct REG_EXP
{
	eExpType				eType;
	FLMUINT				uiMinOccurs;
	FLMUINT				uiMaxOccurs;
	FLMBOOL				bUnlimited;
	FLMBOOL				bQuantified;
	union
	{
		REGEXP_LITERAL			literal;
		CHAR_CLASS				charClass;
		REG_EXP_ALTERNATIVE	alternative;
	} exp;
	REG_EXP_BRANCH *	pBranch;
	REG_EXP *			pNext;
	REG_EXP *			pPrev;
} REG_EXP;

/*****************************************************************************
Desc:	The regular expression class.
*****************************************************************************/
class F_RegExp : public XF_Base
{
public:

	F_RegExp();

	~F_RegExp();

	RCODE setExpression(
		FLMUNICODE *	puzRegExp);

	FLMBOOL testString(
		IF_PosIStream *	pIStream);

private:

	RCODE createRegExp(
		eExpType		eType,
		REG_EXP **	ppExpr);
		
	RCODE addLiteralChar(
		FLMUNICODE	uzChar);

	RCODE addLiteralExpr(
		FLMUNICODE *	puzLiteral,
		FLMUINT			uiNumChars,
		REG_EXP **		ppExpr);

	RCODE saveLiteral( void);

	RCODE createCharCategory(
		CATEGORY_CHARS *	pCategoryChars,
		FLMBOOL				bNegatedCategory,
		CHAR_CLASS *		pCharClass);

	RCODE createCharRange(
		FLMUNICODE		uzLowChar,
		FLMUNICODE		uzHighChar,
		FLMBOOL			bNegatedRange,
		CHAR_CLASS *	pCharClass);

	RCODE createCharList(
		const char *	pszChars,
		FLMUNICODE *	puzChars,
		FLMUINT			uiNumChars,
		FLMBOOL			bNegatedChars,
		CHAR_CLASS *	pCharClass);

	RCODE createCharClass(
		FLMBOOL			bNegatedClass,
		CHAR_CLASS *	pCharClass,
		CHAR_CLASS **	ppNewCharClass);
		
	RCODE parseEscape(
		FLMUNICODE **		ppuzRegExp,
		CHAR_CLASS *		pCharClass,
		FLMUNICODE *		puzChar);

	RCODE parseCharClass(
		FLMUNICODE **	ppuzRegExp,
		CHAR_CLASS *	pCharClass);

	RCODE parseQuantifier(
		FLMUNICODE **	ppuzRegExp);

	RCODE startAlternative( void);

	RCODE endAlternative( void);

	RCODE startNewBranch( void);

	F_Pool				m_Pool;
	REG_EXP_BRANCH		m_topBranch;
	REG_EXP_BRANCH *	m_pCurrBranch;		// Used only when parsing
	FLMUNICODE			m_uzLiteral [256];
	FLMUNICODE *		m_puzLiteral;
	FLMUINT				m_uiMaxLiteralChars;
	FLMUINT				m_uiNumLiteralChars;
};

// Local function prototypes

FSTATIC FLMBOOL isCategory(
	CATEGORY_CHARS **	ppCategoryChars,
	FLMUNICODE **		ppuzRegExp);

FSTATIC FLMBOOL isBlock(
	FLMUNICODE *	puzLowChar,
	FLMUNICODE *	puzHighChar,
	FLMUNICODE **	ppuzRegExp);

/*****************************************************************************
Desc:	Constructor
*****************************************************************************/
F_RegExp::F_RegExp()
{
	m_Pool.poolInit( 256);
	m_topBranch.pParentExpr = NULL;
	m_topBranch.pNextBranch = NULL;
	m_topBranch.pPrevBranch = NULL;
	m_topBranch.pFirstExpr = NULL;
	m_topBranch.pLastExpr = NULL;
	m_pCurrBranch = &m_topBranch;
	m_puzLiteral = &m_uzLiteral [0];
	m_uiMaxLiteralChars = sizeof( m_uzLiteral) / sizeof( FLMUNICODE);
	m_uiNumLiteralChars = 0;
}

/*****************************************************************************
Desc:	Destructor
*****************************************************************************/
F_RegExp::~F_RegExp()
{
	m_Pool.poolFree();
	if (m_puzLiteral != &m_uzLiteral [0])
	{
		f_free( &m_puzLiteral);
	}
}

/*****************************************************************************
Desc:	Skip whitespace in a string.
*****************************************************************************/
FINLINE FLMBOOL isWhitespace(
	FLMUNICODE	uzChar
	)
{
	return( (uzChar == ' ' || uzChar == '\t' ||
			   uzChar == '\n' || uzChar == '\r') ? TRUE : FALSE);
}

/*****************************************************************************
Desc:	Skip whitespace in a string.
*****************************************************************************/
FINLINE FLMUNICODE * skipWhitespace(
	FLMUNICODE *	puzRegExp
	)
{
	while (isWhitespace( *puzRegExp))
	{
		puzRegExp++;
	}
	return( puzRegExp);
}

/*****************************************************************************
Desc:	Create a new regular expression.
*****************************************************************************/
RCODE F_RegExp::createRegExp(
	eExpType		eType,
	REG_EXP **	ppExpr
	)
{
	RCODE			rc = NE_XFLM_OK;
	REG_EXP *	pExpr;
	
	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( REG_EXP),
										(void **)&pExpr)))
	{
		goto Exit;
	}
	*ppExpr = pExpr;
	pExpr->eType = eType;
	if ((pExpr->pPrev = m_pCurrBranch->pLastExpr) != NULL)
	{
		m_pCurrBranch->pLastExpr->pNext = pExpr;
	}
	else
	{
		m_pCurrBranch->pFirstExpr = pExpr;
	}
	m_pCurrBranch->pLastExpr = pExpr;
	pExpr->pBranch = m_pCurrBranch;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Add a literal character to our current literal buffer.
*****************************************************************************/
RCODE F_RegExp::addLiteralChar(
	FLMUNICODE	uzChar
	)
{
	RCODE	rc = NE_XFLM_OK;

	// See if we need to allocate a new buffer.

	if (m_uiNumLiteralChars == m_uiMaxLiteralChars)
	{
		FLMUNICODE *	puzTmp;
		FLMUINT			uiNewMax = m_uiMaxLiteralChars + 128;

		if (RC_BAD( rc = f_alloc( sizeof( FLMUNICODE) * uiNewMax,
									&puzTmp)))
		{
			goto Exit;
		}
		if (m_uiNumLiteralChars)
		{
			f_memcpy( puzTmp, m_puzLiteral,
				sizeof( FLMUNICODE) * m_uiNumLiteralChars);
		}
		if (m_puzLiteral != &m_uzLiteral [0])
		{
			f_free( &m_puzLiteral);
		}
		m_puzLiteral = puzTmp;
		m_uiMaxLiteralChars = uiNewMax;
	}
	m_puzLiteral [m_uiNumLiteralChars] = uzChar;
	m_uiNumLiteralChars++;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Add a literal expression.
*****************************************************************************/
RCODE F_RegExp::addLiteralExpr(
	FLMUNICODE *	puzLiteral,
	FLMUINT			uiNumChars,
	REG_EXP **		ppExpr
	)
{
	RCODE			rc = NE_XFLM_OK;
	REG_EXP *	pTmpExpr = 0;
	
	if (RC_BAD( rc = createRegExp( EXP_LITERAL, &pTmpExpr)))
	{
		goto Exit;
	}

	if (ppExpr)
	{
		*ppExpr = pTmpExpr;
	}
	if (RC_BAD( rc = m_Pool.poolAlloc( uiNumChars * sizeof( FLMUNICODE),
										(void **)&pTmpExpr->exp.literal.puzLiteral)))
	{
		goto Exit;
	}
	f_memcpy( pTmpExpr->exp.literal.puzLiteral, puzLiteral,
					uiNumChars * sizeof( FLMUNICODE));
	pTmpExpr->exp.literal.uiNumChars = uiNumChars;
	
Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Save out our current literal into our expression tree.
*****************************************************************************/
RCODE F_RegExp::saveLiteral( void)
{
	RCODE	rc = NE_XFLM_OK;

	if (m_uiNumLiteralChars)
	{
		if (RC_BAD( rc = addLiteralExpr( m_puzLiteral,
									m_uiNumLiteralChars, NULL)))
		{
			goto Exit;
		}

		// Zero out the literal and start over.

		m_uiNumLiteralChars = 0;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	See if an escape sequence is a category
*****************************************************************************/
FSTATIC FLMBOOL isCategory(
	CATEGORY_CHARS **	ppCategoryChars,
	FLMUNICODE **		ppuzRegExp
	)
{
	FLMBOOL			bIsCategory = FALSE;
	FLMUNICODE *	puzRegExp = *ppuzRegExp;

	// Skip past the 'p' or 'P'

	puzRegExp++;

	// Next character better be a '{'

	if (*puzRegExp != '{')
	{
		goto Exit;
	}
	puzRegExp++;

	switch (*puzRegExp)
	{
		case 'L':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &LCategory;						break;
				case 'u': *ppCategoryChars = &LuCategory;	puzRegExp++;	break;
				case 'l': *ppCategoryChars = &LlCategory;	puzRegExp++;	break;
				case 't': *ppCategoryChars = &LtCategory;	puzRegExp++;	break;
				case 'm': *ppCategoryChars = &LmCategory;	puzRegExp++;	break;
				case 'o': *ppCategoryChars = &LoCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		case 'M':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &MCategory;						break;
				case 'n': *ppCategoryChars = &MnCategory;	puzRegExp++;	break;
				case 'c': *ppCategoryChars = &McCategory;	puzRegExp++;	break;
				case 'e': *ppCategoryChars = &MeCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		case 'N':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &NCategory;						break;
				case 'd': *ppCategoryChars = &NdCategory;	puzRegExp++;	break;
				case 'l': *ppCategoryChars = &NlCategory;	puzRegExp++;	break;
				case 'o': *ppCategoryChars = &NoCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		case 'P':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &PCategory;						break;
				case 'c': *ppCategoryChars = &PcCategory;	puzRegExp++;	break;
				case 'd': *ppCategoryChars = &PdCategory;	puzRegExp++;	break;
				case 's': *ppCategoryChars = &PsCategory;	puzRegExp++;	break;
				case 'e': *ppCategoryChars = &PeCategory;	puzRegExp++;	break;
				case 'i': *ppCategoryChars = &PiCategory;	puzRegExp++;	break;
				case 'f': *ppCategoryChars = &PfCategory;	puzRegExp++;	break;
				case 'o': *ppCategoryChars = &PoCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		case 'Z':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &ZCategory;						break;
				case 's': *ppCategoryChars = &ZsCategory;	puzRegExp++;	break;
				case 'l': *ppCategoryChars = &ZlCategory;	puzRegExp++;	break;
				case 'p': *ppCategoryChars = &ZpCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		case 'S':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &SCategory;						break;
				case 'm': *ppCategoryChars = &SmCategory;	puzRegExp++;	break;
				case 'c': *ppCategoryChars = &ScCategory;	puzRegExp++;	break;
				case 'k': *ppCategoryChars = &SkCategory;	puzRegExp++;	break;
				case 'o': *ppCategoryChars = &SoCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		case 'C':
			puzRegExp++;
			switch (*puzRegExp)
			{
				case '}': *ppCategoryChars = &CCategory;						break;
				case 'c': *ppCategoryChars = &CcCategory;	puzRegExp++;	break;
				case 'f': *ppCategoryChars = &CfCategory;	puzRegExp++;	break;
				case 'o': *ppCategoryChars = &CoCategory;	puzRegExp++;	break;
				case 'n': *ppCategoryChars = &CnCategory;	puzRegExp++;	break;
				default: goto Exit;
			}
			break;

		default:
			goto Exit;
	}

	// The last letter better be a '}'

	if (*puzRegExp != '}')
	{
		goto Exit;
	}
	puzRegExp++;
	bIsCategory = TRUE;

Exit:

	// Only move the pointer forward if it is, in fact, a category

	if (bIsCategory)
	{
		*ppuzRegExp = puzRegExp;
	}
	return( bIsCategory);
}

/*****************************************************************************
Desc:	See if an escape sequence is a block range of characters
*****************************************************************************/
FSTATIC FLMBOOL isBlock(
	FLMUNICODE *	puzLowChar,
	FLMUNICODE *	puzHighChar,
	FLMUNICODE **	ppuzRegExp
	)
{
	FLMBOOL			bIsBlock = FALSE;
	FLMUINT			uiLoop;
	FLMUNICODE *	puzRegExp = *ppuzRegExp;
	FLMUNICODE *	puzBlockName;
	FLMUNICODE *	puzSaveBlockName;
	const char *	pszBlockName;

	// Skip past the 'p' or 'P'

	puzRegExp++;

	// Next three characters better be '{Is'

	if (*puzRegExp != '{')
	{
		goto Exit;
	}
	puzRegExp++;

	if (*puzRegExp != 'I')
	{
		goto Exit;
	}
	puzRegExp++;

	if (*puzRegExp != 's')
	{
		goto Exit;
	}
	puzRegExp++;

	puzSaveBlockName = puzRegExp;
	uiLoop = 0;
	for (uiLoop = 0;;uiLoop++)
	{
		if ((pszBlockName = FlmBlockCharRanges [uiLoop].pszBlockName) == NULL)
		{
			goto Exit;
		}

		// Compare the name

		puzBlockName = puzSaveBlockName;
		while (*pszBlockName && *puzBlockName != '}')
		{
			if ((FLMUNICODE)(*pszBlockName) != *puzBlockName)
			{
				break;
			}
			pszBlockName++;
			puzBlockName++;
		}

		if (*pszBlockName == 0 && *puzBlockName == '}')
		{
			puzRegExp = puzBlockName + 1;
			bIsBlock = TRUE;
			*puzLowChar = FlmBlockCharRanges [uiLoop].uzLowChar;
			*puzHighChar = FlmBlockCharRanges [uiLoop].uzLowChar;
			bIsBlock = TRUE;
			break;
		}
	}

Exit:

	// Only move the pointer forward if it is, in fact, a category

	if (bIsBlock)
	{
		*ppuzRegExp = puzRegExp;
	}
	return( bIsBlock);
}

/*****************************************************************************
Desc:	Create a character category.
*****************************************************************************/
RCODE F_RegExp::createCharCategory(
	CATEGORY_CHARS *	pCategoryChars,
	FLMBOOL				bNegatedCategory,
	CHAR_CLASS *		pCharClass
	)
{
	RCODE					rc = NE_XFLM_OK;
	CHAR_CATEGORY *	pCharCategory;

	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( CHAR_CATEGORY),
							(void **)&pCharCategory)))
	{
		goto Exit;
	}
	pCharCategory->bNegatedCategory = bNegatedCategory;
	pCharCategory->pCategoryChars = pCategoryChars;
	
	// Link at end of list of character categories

	if ((pCharCategory->pPrev = pCharClass->pLastCharCategory) != NULL)
	{
		pCharCategory->pPrev->pNext = pCharCategory;
	}
	else
	{
		pCharClass->pFirstCharCategory = pCharCategory;
	}
	pCharClass->pLastCharCategory = pCharCategory;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Create a character range.
*****************************************************************************/
RCODE F_RegExp::createCharRange(
	FLMUNICODE		uzLowChar,
	FLMUNICODE		uzHighChar,
	FLMBOOL			bNegatedRange,
	CHAR_CLASS *	pCharClass
	)
{
	RCODE				rc = NE_XFLM_OK;
	CHAR_RANGE *	pCharRange;

	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( CHAR_RANGE),
							(void **)&pCharRange)))
	{
		goto Exit;
	}
	pCharRange->bNegatedRange = bNegatedRange;
	pCharRange->uzLowChar = uzLowChar;
	pCharRange->uzHighChar = uzHighChar;
	
	// Link at end of list of character ranges

	if ((pCharRange->pPrev = pCharClass->pLastCharRange) != NULL)
	{
		pCharRange->pPrev->pNext = pCharRange;
	}
	else
	{
		pCharClass->pFirstCharRange = pCharRange;
	}
	pCharClass->pLastCharRange = pCharRange;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Create a character list.
*****************************************************************************/
RCODE F_RegExp::createCharList(
	const char *	pszChars,
	FLMUNICODE *	puzChars,
	FLMUINT			uiNumChars,
	FLMBOOL			bNegatedChars,
	CHAR_CLASS *	pCharClass)
{
	RCODE				rc = NE_XFLM_OK;
	CHAR_LIST *		pCharList;

	// Allocate the CHAR_LIST structure.

	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( CHAR_LIST),
							(void **)&pCharList)))
	{
		goto Exit;
	}
	pCharList->bNegatedChars = bNegatedChars;

	// Allocate an array for the characters.

	if (RC_BAD( rc = m_Pool.poolAlloc( sizeof( FLMUNICODE) * uiNumChars,
								(void **)&pCharList->puzChars)))
	{
		goto Exit;
	}
	pCharList->uiNumChars = uiNumChars;

	// Copy the characters from pszChars or puzChars into pCharList->puzChars

	if (pszChars)
	{
		flmAssert( !puzChars);
		puzChars = pCharList->puzChars;
		while (*pszChars)
		{
			*puzChars = (FLMUNICODE)(*pszChars);
			puzChars++;
			pszChars++;
		}
	}
	else
	{
		f_memcpy( pCharList->puzChars, puzChars,
				sizeof( FLMUNICODE) * uiNumChars);
	}

	// Link at end of list of character lists

	if ((pCharList->pPrev = pCharClass->pLastCharList) != NULL)
	{
		pCharList->pPrev->pNext = pCharList;
	}
	else
	{
		pCharClass->pFirstCharList = pCharList;
	}
	pCharClass->pLastCharList = pCharList;

Exit:

	return( rc);
}
				
/*****************************************************************************
Desc:	Create a sub-character class
*****************************************************************************/
RCODE F_RegExp::createCharClass(
	FLMBOOL			bNegatedClass,
	CHAR_CLASS *	pCharClass,
	CHAR_CLASS **	ppNewCharClass
	)
{
	RCODE				rc = NE_XFLM_OK;
	CHAR_CLASS *	pNewCharClass;

	// Allocate the CHAR_CLASS structure.

	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( CHAR_CLASS),
							(void **)&pNewCharClass)))
	{
		goto Exit;
	}
	*ppNewCharClass = pNewCharClass;
	pNewCharClass->bNegatedClass = bNegatedClass;

	// Link at end of list of character classes

	if ((pNewCharClass->pPrev = pCharClass->pLastCharClass) != NULL)
	{
		pNewCharClass->pPrev->pNext = pCharClass;
	}
	else
	{
		pNewCharClass->pFirstCharClass = pCharClass;
	}
	pCharClass->pLastCharClass = pNewCharClass;

Exit:

	return( rc);
}
				
/*****************************************************************************
Desc:	Parse an escape sequence.
*****************************************************************************/
RCODE F_RegExp::parseEscape(
	FLMUNICODE **		ppuzRegExp,
	CHAR_CLASS *		pCharClass,
	FLMUNICODE *		puzChar
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzRegExp = *ppuzRegExp;
	CHAR_CLASS *		pNewCharClass;
	CATEGORY_CHARS *	pCategoryChars;
	FLMUNICODE			uzLowChar;
	FLMUNICODE			uzHighChar;
	FLMBOOL				bNegated;
	REG_EXP *			pTmpExpr;

	*puzChar = 0;

	// Skip past the '\'

	puzRegExp++;

	switch (*puzRegExp)
	{
		case 'p':
		case 'P':
			bNegated = (*puzRegExp == 'P') ? TRUE : FALSE;
			if (isCategory( &pCategoryChars, &puzRegExp))
			{
				if (!pCharClass)
				{
					if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
					{
						goto Exit;
					}
					pCharClass = &pTmpExpr->exp.charClass;
				}
				if (RC_BAD( rc = createCharCategory( pCategoryChars, bNegated,
											pCharClass)))
				{
					goto Exit;
				}
			}
			else if (isBlock( &uzLowChar, &uzHighChar, &puzRegExp))
			{
				if (!pCharClass)
				{
					if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
					{
						goto Exit;
					}
					pCharClass = &pTmpExpr->exp.charClass;
				}
				if (RC_BAD( rc = createCharRange( uzLowChar, uzHighChar,
										bNegated, pCharClass)))
				{
					goto Exit;
				}
			}
			else
			{

				// Treat as a regular character.

				*puzChar = *puzRegExp;
				puzRegExp++;
			}
			break;

		case 's':
		case 'S':
			bNegated = (*puzRegExp == 'S') ? TRUE : FALSE;
			puzRegExp++;
			if (!pCharClass)
			{
				if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
				{
					goto Exit;
				}
				pCharClass = &pTmpExpr->exp.charClass;
			}
			if (RC_BAD( rc = createCharList( " \t\n\r", NULL, 4,
										bNegated, pCharClass)))
			{
				goto Exit;
			}
			break;

		case 'd':
		case 'D':

			// The same as {Nd} category

			bNegated = (*puzRegExp == 'D') ? TRUE : FALSE;
			puzRegExp++;
			if (!pCharClass)
			{
				if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
				{
					goto Exit;
				}
				pCharClass = &pTmpExpr->exp.charClass;
			}
			if (RC_BAD( rc = createCharCategory( &NdCategory,
														 bNegated, pCharClass)))
			{
				goto Exit;
			}
			break;

		case 'w':
		case 'W':

			// Create a sub-character class that excludes the
			// categories {P} - punctuation, {Z} - separators, and {C} - others
			// NOTE: bNegated should be set to TRUE for lowercase 'w', unlike
			// others above where negated flag is TRUE for uppercase
			// character.  This is because we are trying to create a class
			// that is everything EXCEPT the three character categories
			// mentioned.  In the case of 'W', it should include those
			// three categories - just the opposite of 'w'.

			bNegated = (*puzRegExp == 'w') ? TRUE : FALSE;
			puzRegExp++;
			if (!pCharClass)
			{
				if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
				{
					goto Exit;
				}
				pCharClass = &pTmpExpr->exp.charClass;
			}
			if (RC_BAD( rc = createCharClass( bNegated, pCharClass,
										&pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharCategory( &PCategory,
										FALSE, pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharCategory( &ZCategory,
										FALSE, pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharCategory( &CCategory,
										FALSE, pNewCharClass)))
			{
				goto Exit;
			}
			break;

		case 'c':
		case 'C':

			// NameChar and '.', '_', '-', ':'

			bNegated = (*puzRegExp == 'C') ? TRUE : FALSE;
			puzRegExp++;
			if (!pCharClass)
			{
				if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
				{
					goto Exit;
				}
				pCharClass = &pTmpExpr->exp.charClass;
			}
			if (RC_BAD( rc = createCharClass( bNegated, pCharClass, &pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharCategory( &NameCharCategory, FALSE,
									pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharList( "._-:", NULL, 4, FALSE, pNewCharClass)))
			{
				goto Exit;
			}
			break;

		case 'i':
		case 'I':

			// Letter and '_', ':'

			bNegated = (*puzRegExp == 'I') ? TRUE : FALSE;
			puzRegExp++;
			if (!pCharClass)
			{
				if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
				{
					goto Exit;
				}
				pCharClass = &pTmpExpr->exp.charClass;
			}
			if (RC_BAD( rc = createCharClass( bNegated, pCharClass, &pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharCategory( &LetterCategory, FALSE,
									pNewCharClass)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = createCharList( "_:", NULL, 2, FALSE, pNewCharClass)))
			{
				goto Exit;
			}
			break;

		case 'n':
			*puzChar = 0xA;
			puzRegExp++;
			break;

		case 'r':
			*puzChar = 0xD;
			puzRegExp++;
			break;

		case 't':
			*puzChar = 0x9;
			puzRegExp++;
			break;

		default:

			*puzChar = *puzRegExp;
			puzRegExp++;
			break;
	}

Exit:

	*ppuzRegExp = puzRegExp;
	return( rc);
}

/*****************************************************************************
Desc:	Parse a character class - [xxxxx].
*****************************************************************************/
RCODE F_RegExp::parseCharClass(
	FLMUNICODE **	ppuzRegExp,
	CHAR_CLASS *	pCharClass
	)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE *		puzRegExp = *ppuzRegExp;
	FLMBOOL				bAtBeginning;
	FLMBOOL				bHaveDash;
	FLMUNICODE			uzChar;

	flmAssert( *puzRegExp == '[');

	// skip past the '['

	puzRegExp++;

	// Save whatever literal expression may have built up.

	if (RC_BAD( rc = saveLiteral()))
	{
		goto Exit;
	}

	bAtBeginning = TRUE;
	bHaveDash = FALSE;
	for (;;)
	{
		switch (*puzRegExp)
		{
			case 0:
				rc = RC_SET( NE_XFLM_UNEXPECTED_END_OF_EXPR);
				goto Exit;

			case '^':

				// This is only the negation character if it comes immediately
				// after the opening '['.  Otherwise, it is a regular character.

				if (bAtBeginning && !pCharClass->bNegatedClass)
				{
					pCharClass->bNegatedClass = TRUE;
					puzRegExp++;
				}
				else
				{
					goto Save_Char;
				}
				break;

			case '-':

				// This is NOT a range operator if it comes immediately after
				// the opening '[' or right before the closing ']'.  In those
				// two cases, it is a regular character.

				if (bAtBeginning || *(puzRegExp + 1) == ']')
				{
					goto Save_Char;
				}
				else if (!m_uiNumLiteralChars)
				{
					// The dash doesn't have a preceding character we can use
					// as the beginning character of the range.

					rc = RC_SET( NE_XFLM_UNESCAPED_METACHAR);
					goto Exit;
				}
				else
				{
					bHaveDash = TRUE;
					puzRegExp++;
				}
				break;

			case '\\':
			
				if (RC_BAD( rc = parseEscape( &puzRegExp, pCharClass, &uzChar)))
				{
					goto Exit;
				}
				if (uzChar)
				{
					goto Save_Char;
				}
				bAtBeginning = FALSE;

				break;

			case ']':

				// If it comes right after the opening '[', or after the '^' when it
				// is used as a negation (see above), it is a regular character.
				// In both of those cases, bAtBeginning will still be TRUE.
				// Otherwise, it is the end of the character class.

				if (bAtBeginning)
				{
					goto Save_Char;
				}
				goto End_Of_Expr;

			case '[':

				// If it comes right after a '-', it represents the beginning of
				// a subtraction group.

				if (!bHaveDash)
				{
					goto Save_Char;
				}

				// Won't be at the beginning in this case

				flmAssert( !bAtBeginning);

				// Had the dash character, toggle flag back to FALSE.

				bHaveDash = FALSE;

				// Before calling self recursively, need to clear out
				// any characters we have gathered up so far.

				if (m_uiNumLiteralChars)
				{
					if (RC_BAD( rc = createCharList( NULL, m_puzLiteral,
											m_uiNumLiteralChars, FALSE,
											pCharClass)))
					{
						goto Exit;
					}
					m_uiNumLiteralChars = 0;
				}

				if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( CHAR_CLASS),
													(void **)&pCharClass->pSubtractionClass)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = parseCharClass( &puzRegExp,
												pCharClass->pSubtractionClass)))
				{
					goto Exit;
				}

				// Next character must be a ']' to end this character class

				if (*puzRegExp == ']')
				{
					goto End_Of_Expr;
				}
				rc = (RCODE)((*puzRegExp)
								 ? RC_SET( NE_XFLM_ILLEGAL_CLASS_SUBTRACTION)
								 : RC_SET( NE_XFLM_UNEXPECTED_END_OF_EXPR));
				goto Exit;

			default:

Save_Char:

				if (!bHaveDash)
				{
					if (RC_BAD( rc = addLiteralChar( *puzRegExp)))
					{
						goto Exit;
					}
				}
				else
				{
					FLMUNICODE		uzLowChar;

					// Reset the bHaveDash flag.

					bHaveDash = FALSE;
									
					uzLowChar = m_puzLiteral [m_uiNumLiteralChars - 1];
					if (*puzRegExp < uzLowChar)
					{
						rc = RC_SET( NE_XFLM_ILLEGAL_CHAR_RANGE_IN_EXPR);
						goto Exit;
					}

					// No need to do anything if the character is equal to
					// the last one.  i.e., they are doing a range like A-A

					else if (*puzRegExp > uzLowChar)
					{

						// Save off the characters we have gathered so far,
						// except for the last one, which will be the beginning
						// of our range.

						if (m_uiNumLiteralChars > 1)
						{
							if (RC_BAD( rc = createCharList( NULL, m_puzLiteral,
														m_uiNumLiteralChars - 1, FALSE,
														pCharClass)))
							{
								goto Exit;
							}
						}

						// Need to zero out number of characters so that if there
						// is another dash, it will be reported as an unescaped
						// metacharacter.

						m_uiNumLiteralChars = 0;

						// Create a range.

						if (RC_BAD( rc = createCharRange( uzLowChar, *puzRegExp,
													FALSE, pCharClass)))
						{
							goto Exit;
						}
					}
				}
				bAtBeginning = FALSE;
				puzRegExp++;
				break;
		}
	}

End_Of_Expr:

	// Keep the final set of characters we may have
	// gathered.

	if (m_uiNumLiteralChars)
	{
		if (RC_BAD( rc = createCharList( NULL, m_puzLiteral,
									m_uiNumLiteralChars, FALSE, pCharClass)))
		{
			goto Exit;
		}
		m_uiNumLiteralChars = 0;
	}

	// Skip past the ']'

	puzRegExp++;

Exit:

	*ppuzRegExp = puzRegExp;
	return( rc);
}

/*****************************************************************************
Desc:	Parse a quantifier expression.
		All of the following forms are allowed:
		{3} - exactly 3
		{,4} - same as {0,4}
		{3,} - 3 to unlimited
		{3,5} - min of 3, max of 5
		+ - same as {1,}
		* - same as {0,}
		? - same as {0,1}
*****************************************************************************/
RCODE F_RegExp::parseQuantifier(
	FLMUNICODE **	ppuzRegExp)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUNICODE *	puzRegExp = *ppuzRegExp;
	FLMUINT			uiMin;
	FLMUINT			uiMax;
	FLMBOOL			bUnlimited;
	REG_EXP *		pTmpExpr;

	if (!m_uiNumLiteralChars &&
		 (!m_pCurrBranch->pLastExpr ||
		  m_pCurrBranch->pLastExpr->bQuantified))
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_QUANTIFIER);
		goto Exit;
	}

	// Skip the first character

	if (*puzRegExp == '?')
	{
		uiMin = 0;
		uiMax = 1;
		bUnlimited = FALSE;

		// Skip past the '?'

		puzRegExp++;
	}
	else if (*puzRegExp == '*')
	{
		uiMin = 0;
		uiMax = 0;
		bUnlimited = TRUE;

		// Skip past the '*'

		puzRegExp++;
	}
	else if (*puzRegExp == '+')
	{
		uiMin = 1;
		uiMax = 0;
		bUnlimited = TRUE;

		// Skip past the '+'

		puzRegExp++;
	}
	else
	{

		// Only thing left better be a left brace

		flmAssert( *puzRegExp == '{');
		
		// Skip past the left brace

		puzRegExp++;

		// Skip any white space

		puzRegExp = skipWhitespace( puzRegExp);

		// Get the first number, if any

		uiMin = 0;
		while (*puzRegExp >= '0' && *puzRegExp <= '9')
		{
			uiMin *= 10;
			uiMin += (FLMUINT)(*puzRegExp - '0');
			puzRegExp++;
		}

		// Skip any whitespace that comes after the number.

		puzRegExp = skipWhitespace( puzRegExp);

		// Better have landed on a comma or right brace

		if (*puzRegExp == ',')
		{
			puzRegExp++;

			// Skip any whitespace that comes after the comma

			puzRegExp = skipWhitespace( puzRegExp);
		}
		else if (*puzRegExp == '}')
		{
			if (!uiMin)
			{
				rc = RC_SET( NE_XFLM_ILLEGAL_QUANTIFIER);
				goto Exit;
			}
			uiMax = uiMin;
			bUnlimited = FALSE;
		}
		else
		{
			rc = RC_SET( NE_XFLM_ILLEGAL_MIN_COUNT);
			goto Exit;
		}

		uiMax = 0;
		bUnlimited = TRUE;

		// Get the next number, if any

		while (*puzRegExp >= '0' && *puzRegExp <= '9')
		{
			uiMax *= 10;
			uiMax += (FLMUINT)(*puzRegExp - '0');
			bUnlimited = FALSE;
			puzRegExp++;
		}

		// Skip any whitespace that comes after the number.

		puzRegExp = skipWhitespace( puzRegExp);

		// Better have landed on a right brace

		if (*puzRegExp != '}')
		{
			rc = (RCODE)((*puzRegExp)
							 ? RC_SET( NE_XFLM_ILLEGAL_MAX_COUNT)
							 : RC_SET( NE_XFLM_UNEXPECTED_END_OF_EXPR));
			goto Exit;
		}

		// Got the '}', see if min and max are legal.

		puzRegExp++;
		if (!bUnlimited && (!uiMax || uiMax < uiMin))
		{
			rc = RC_SET( NE_XFLM_ILLEGAL_MAX_COUNT);
			goto Exit;
		}
	}

	// If we have a literal, create two expressions.  First
	// expression will be all but the last character of the
	// literal.  Second expression will be the one character
	// literal with a count.

	if (m_uiNumLiteralChars)
	{
		if (m_uiNumLiteralChars > 1)
		{
			if (RC_BAD( rc = addLiteralExpr( m_puzLiteral,
										m_uiNumLiteralChars - 1, NULL)))
			{
				goto Exit;
			}
		}
		if (RC_BAD( rc = addLiteralExpr(
								&m_puzLiteral [m_uiNumLiteralChars - 1], 1,
								&pTmpExpr)))
		{
			goto Exit;
		}

		// Zero out the literal and start over.

		m_uiNumLiteralChars = 0;
	}
	else
	{
		pTmpExpr = m_pCurrBranch->pLastExpr;
	}
	flmAssert( pTmpExpr);
	pTmpExpr->uiMinOccurs = uiMin;
	pTmpExpr->uiMaxOccurs = uiMax;
	pTmpExpr->bUnlimited = bUnlimited;
	pTmpExpr->bQuantified = TRUE;

Exit:

	*ppuzRegExp = puzRegExp;
	return( rc);
}

/*****************************************************************************
Desc:	Start an alternative - Called when we hit a left paren.
*****************************************************************************/
RCODE F_RegExp::startAlternative( void)
{
	RCODE			rc = NE_XFLM_OK;
	REG_EXP *	pTmpExpr = 0;

	// If we were gathering up a literal, save it out.

	if (RC_BAD( rc = saveLiteral()))
	{
		goto Exit;
	}

	// Start a new alternative expression node

	if (RC_BAD( rc = createRegExp( EXP_ALTERNATIVES, &pTmpExpr)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( REG_EXP_BRANCH),
							(void **)&pTmpExpr->exp.alternative.pFirstBranch)))
	{
		goto Exit;
	}
	pTmpExpr->exp.alternative.pLastBranch =
				pTmpExpr->exp.alternative.pFirstBranch;
	m_pCurrBranch = pTmpExpr->exp.alternative.pFirstBranch;
	m_pCurrBranch->pParentExpr = pTmpExpr;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	End an alternative - Called when we hit a right paren.
*****************************************************************************/
RCODE F_RegExp::endAlternative( void)
{
	RCODE			rc = NE_XFLM_OK;
	REG_EXP *	pParentExpr;
	REG_EXP *	pTmpExpr;

	// If the current branch doesn't have a parent
	// expression, this is an illegal unescaped right paren.

	if ((pParentExpr = m_pCurrBranch->pParentExpr) == NULL)
	{
		rc = RC_SET( NE_XFLM_UNESCAPED_METACHAR);
		goto Exit;
	}
	flmAssert( pParentExpr->eType == EXP_ALTERNATIVES);

	// If we were gathering up a literal, save it out.

	if (RC_BAD( rc = saveLiteral()))
	{
		goto Exit;
	}

	// Make sure the current branch isn't empty

	if (!m_pCurrBranch->pFirstExpr)
	{
		rc = RC_SET( NE_XFLM_EMPTY_BRANCH_IN_EXPR);
		goto Exit;
	}

	// If there is only one alternative, link these
	// nodes right in where the parent expression would
	// have gone.  This is not strictly necessary, because
	// the processor can handle only one alternative, but
	// it may save processing time in the end.

	if (pParentExpr->exp.alternative.pFirstBranch ==
		 pParentExpr->exp.alternative.pLastBranch)
	{
		if ((m_pCurrBranch->pFirstExpr->pPrev = pParentExpr->pPrev) != NULL)
		{
			pParentExpr->pPrev->pNext = m_pCurrBranch->pFirstExpr;
		}
		else
		{
			pParentExpr->pBranch->pFirstExpr = m_pCurrBranch->pFirstExpr;
		}

		// Need to alter the branch pointed to by each of the
		// expressions in this list.

		pTmpExpr = m_pCurrBranch->pFirstExpr;
		while (pTmpExpr)
		{
			pTmpExpr->pBranch = pParentExpr->pBranch;
			pTmpExpr = pTmpExpr->pNext;
		}
		pParentExpr->pBranch->pLastExpr = m_pCurrBranch->pLastExpr;
	}

	// Go back to the parent branch

	m_pCurrBranch = pParentExpr->pBranch;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Start a new branch of an alternatives list.  This is called when
		we hit the '|' character.
*****************************************************************************/
RCODE F_RegExp::startNewBranch( void)
{
	RCODE					rc = NE_XFLM_OK;
	REG_EXP *			pTmpExpr;
	REG_EXP_BRANCH *	pTmpBranch;

	// If we were gathering up a literal, save it out.

	if (RC_BAD( rc = saveLiteral()))
	{
		goto Exit;
	}

	// Make sure the current branch isn't empty

	if (!m_pCurrBranch->pFirstExpr)
	{
		rc = RC_SET( NE_XFLM_EMPTY_BRANCH_IN_EXPR);
		goto Exit;
	}

	// Create a new branch to link to current branch.

	if (RC_BAD( rc = m_Pool.poolCalloc( sizeof( REG_EXP_BRANCH),
										(void **)&pTmpBranch)))
	{
		goto Exit;
	}

	// Link this branch after the current branch
	
	pTmpBranch->pPrevBranch = m_pCurrBranch;
	m_pCurrBranch->pNextBranch = pTmpBranch;

	// If current branch has a parent, the parent should be
	// an alternative, and it's last branch should now point
	// to this new branch.

	if ((pTmpBranch->pParentExpr = m_pCurrBranch->pParentExpr) != NULL)
	{
		pTmpExpr = pTmpBranch->pParentExpr;
		flmAssert( pTmpExpr->eType == EXP_ALTERNATIVES);
		pTmpExpr->exp.alternative.pLastBranch = pTmpBranch;
	}

	// Current branch should now become this newly created branch

	m_pCurrBranch = pTmpBranch;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Set a regular expression.  Parse the expression, turning it into
		constructs that can be used to test strings more easily.
*****************************************************************************/
RCODE F_RegExp::setExpression(
	FLMUNICODE *	puzRegExp
	)
{
	RCODE			rc = NE_XFLM_OK;
	REG_EXP *	pTmpExpr = 0;
	FLMUNICODE	uzChar;

	while (*puzRegExp)
	{
		switch (*puzRegExp)
		{
			case '[':
				if (RC_BAD( rc = createRegExp( EXP_CHAR_CLASS, &pTmpExpr)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = parseCharClass( &puzRegExp, &pTmpExpr->exp.charClass)))
				{
					goto Exit;
				}
				break;

			case '\\':
				if (RC_BAD( rc = parseEscape( &puzRegExp, NULL, &uzChar)))
				{
					goto Exit;
				}
				if (uzChar)
				{
					if (RC_BAD( rc = addLiteralChar( uzChar)))
					{
						goto Exit;
					}
				}
				break;

			case '|':
				if (RC_BAD( rc = startNewBranch()))
				{
					goto Exit;
				}

				// Skip past the '|'

				puzRegExp++;
				break;

			case '(':
				if (RC_BAD( rc = startAlternative()))
				{
					goto Exit;
				}

				// Skip past the '('

				puzRegExp++;
				break;

			case ')':
				if (RC_BAD( rc = endAlternative()))
				{
					goto Exit;
				}

				// Skip past the ')'

				puzRegExp++;
				break;

			case '{':
			case '+':
			case '*':
			case '?':
				if (RC_BAD( rc = parseQuantifier( &puzRegExp)))
				{
					goto Exit;
				}
				break;

			case '.':
			case '^':
			case ']':
			case '}':
				rc = RC_SET( NE_XFLM_UNESCAPED_METACHAR);
				goto Exit;

			default:

				// Add character to the literal expression we
				// are saving up.

				if (RC_BAD( rc = addLiteralChar( *puzRegExp)))
				{
					goto Exit;
				}
				puzRegExp++;
				break;
		}
	}

	// Output the last literal, if any

	if (RC_BAD( rc = saveLiteral()))
	{
		goto Exit;
	}

	// Make sure we are not nested in parens.

	if (m_pCurrBranch->pParentExpr)
	{
		rc = RC_SET( NE_XFLM_UNEXPECTED_END_OF_EXPR);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Test a string to see if it matches the regular expression.
*****************************************************************************/
FLMBOOL F_RegExp::testString(
	IF_PosIStream *	// pIStream
	)
{
	return( FALSE);
}
