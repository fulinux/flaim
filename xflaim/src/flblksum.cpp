//------------------------------------------------------------------------------
// Desc:	This file contains the routine which calculates a block checksum.
//
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flblksum.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/********************************************************************
Desc:	Calculate the checksum for a block.  NOTE: This is ALWAYS done
		on the raw image that will be written to disk.  This means
		that if the block needs to be converted before writing it out,
		it should be done before calculating the checksum.
*********************************************************************/
FLMUINT32 calcBlkCRC(
	F_BLK_HDR *		pBlkHdr,
	FLMUINT			uiBlkEnd)
{
	FLMUINT32	ui32SaveCRC;
	FLMUINT		uiAdds;
	FLMUINT		uiXORs;
	FLMBYTE *	pucBlkPtr;

	// Calculate CRC on everything except for the ui32BlkCRC value.
	// To do this, we temporarily change it to zero.  The saved
	// value will be restored after calculating the CRC.

	ui32SaveCRC = pBlkHdr->ui32BlkCRC;
	pBlkHdr->ui32BlkCRC = 0;
	uiAdds = 0;
	uiXORs = 0;
	pucBlkPtr = (FLMBYTE *)pBlkHdr;

#if defined( FLM_NLM) || (defined( FLM_WIN) && !defined( FLM_64BIT))

	FastBlockCheckSum( pucBlkPtr, &uiAdds, &uiXORs,
		(unsigned long)uiBlkEnd);

#else
	
	FLMBYTE *		pucCur = pucBlkPtr;
	FLMBYTE *		pucEnd = pucCur + uiBlkEnd;

	while( pucCur < pucEnd)	
	{
		uiAdds += *pucCur;
		uiXORs ^= *pucCur++;
	}

	uiAdds &= 0xFF;
#endif

	// Restore the CRC that was in the block.

	pBlkHdr->ui32BlkCRC = ui32SaveCRC;
	return( (FLMUINT32)((uiAdds << 16) + uiXORs));
}

/********************************************************************
Desc:	Calculate the CRC for the database header.
*********************************************************************/
FLMUINT32 calcDbHdrCRC(
	XFLM_DB_HDR *	pDbHdr)
{
	FLMUINT32	ui32SaveCRC;
	FLMUINT		uiAdds;
	FLMUINT		uiXORs;
	FLMBYTE *	pucHdr;

	// Checksum everything except for the ui32HdrCRC value.

	ui32SaveCRC = pDbHdr->ui32HdrCRC;
	pDbHdr->ui32HdrCRC = 0;

	uiAdds = 0; 
	uiXORs = 0;
	pucHdr = (FLMBYTE *)pDbHdr;

#if defined( FLM_NLM) || (defined( FLM_WIN) && !defined( FLM_64BIT))

	FastBlockCheckSum( pucHdr, &uiAdds, &uiXORs,
		(unsigned long)sizeof( XFLM_DB_HDR));

#else
	FLMBYTE *		pucCur = pucHdr;
	FLMBYTE *		pucEnd = pucHdr + sizeof( XFLM_DB_HDR);

	while( pucCur < pucEnd)	
	{
		uiAdds += *pucCur;
		uiXORs ^= *pucCur++;
	}

	uiAdds &= 0xFF;
#endif

	// Restore the checksum that was in the header

	pDbHdr->ui32HdrCRC = ui32SaveCRC;
	return( (FLMUINT32)((uiAdds << 16) + uiXORs));
}
