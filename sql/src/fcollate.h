//------------------------------------------------------------------------------
// Desc:	Header for collation routines
//
// Tabs:	3
//
//		Copyright (c) 1991-1992, 1994-2000, 2002-2006 Novell, Inc.
//		All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

#ifndef FCOLLATE_H
#define FCOLLATE_H

#define TRUNCATED_FLAG					0x8000
#define EXCLUSIVE_LT_FLAG				0x4000
#define EXCLUSIVE_GT_FLAG				0x2000
#define SEARCH_KEY_FLAG					0x1000

#define KEY_COMPONENT_LENGTH_MASK	0x0FFF
#define KEY_LOW_VALUE					0x0FFE
#define KEY_HIGH_VALUE					0x0FFF

FINLINE FLMBOOL isKeyComponentLTExclusive(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & EXCLUSIVE_LT_FLAG) ? TRUE : FALSE);
}

FINLINE FLMBOOL isKeyComponentGTExclusive(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & EXCLUSIVE_GT_FLAG) ? TRUE : FALSE);
}

FINLINE FLMBOOL isKeyComponentTruncated(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & TRUNCATED_FLAG) ? TRUE : FALSE);
}

FINLINE FLMBOOL isSearchKeyComponent(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & SEARCH_KEY_FLAG) ? TRUE : FALSE);
}

FINLINE FLMUINT getKeyComponentLength(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FLMUINT)(FB2UW( pucKeyComponent)) & KEY_COMPONENT_LENGTH_MASK);
}

RCODE flmColText2StorageText(
	const FLMBYTE *	pucColStr,
	FLMUINT				uiColStrLen,
	FLMBYTE *			pucStorageBuf,
	FLMUINT *			puiStorageLen,
	FLMUINT	   		uiLang,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbFirstSubstring);
	
RCODE flmUTF8ToColText(
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucCollatedStr,
	FLMUINT *			puiCollatedStrLen,
	FLMBOOL  			bCaseInsensitive,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiCaseLen,
	FLMUINT				uiLanguage,
	FLMUINT				uiCharLimit,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bDataTruncated,
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL *			pbDataTruncated);

#endif
