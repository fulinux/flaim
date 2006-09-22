//------------------------------------------------------------------------------
// Desc:	Controlled Cryptographic Services (CCS) interface
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
// $Id: f_nici.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef F_CCS_H
#define F_CCS_H

RCODE flmAllocCCS(
	IF_CCS **		ppCCS);

/****************************************************************************
Desc:	Controlled Cryptographic Services (CCS) interface
****************************************************************************/
class IF_CCS : public F_Object
{
public:

	virtual ~IF_CCS()
	{
	}

	virtual RCODE init(
		FLMBOOL				bKeyIsWrappingKey,
		FLMUINT				uiAlgType) = 0;

	virtual RCODE generateEncryptionKey(
		FLMUINT				uiEncKeySize) = 0;

	virtual RCODE generateWrappingKey(
		FLMUINT				uiEncKeySize) = 0;

	virtual RCODE encryptToStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV = NULL) = 0;

	virtual RCODE decryptFromStore(
		FLMBYTE *			pucIn,
		FLMUINT				uiInLen,
		FLMBYTE *			pucOut,
		FLMUINT *			puiOutLen,
		FLMBYTE *			pucIV = NULL) = 0;

	virtual RCODE getKeyToStore(
		FLMBYTE **			ppucKeyInfo,
		FLMUINT32 *			pui32BufLen,
		FLMBYTE *			pszEncKeyPasswd = NULL,
		IF_CCS *				pWrappingCcs = NULL) = 0;

	virtual RCODE setKeyFromStore(
		FLMBYTE *			pucKeyInfo,
		FLMBYTE *			pszEncKeyPasswd,
		IF_CCS *				pWrappingCcs) = 0;
		
	virtual FLMUINT getIVLen( void) = 0;
	
	virtual RCODE generateIV(
		FLMUINT				uiIVLen,
		FLMBYTE *			pucIV) = 0;
};

#endif // F_CCS_H
