//-------------------------------------------------------------------------
// Desc:	Index key building and comparison routines.
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
// $Id: flkeys.cpp 12263 2006-01-19 14:43:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~***********************************************************************
Desc:	Given an input key tree a FLAIM collated key will be built and returned
		to the user.
****************************************************************************/
RCODE FlmKeyBuild(
	HFDB			hDb,
	FLMUINT		uiIxNum,
	FLMUINT		uiContainer,
	FlmRecord *	pRecord,
	FLMUINT		uiFlag,
	FLMBYTE * 	pKeyBuf,
	FLMUINT *	puiKeyLenRV)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	IXD_p			pIxd;
	FLMBOOL		bImplicitTrans = FALSE;

	if( RC_OK( rc = fdbInit( pDb, FLM_READ_TRANS,
										TRUE, 0, &bImplicitTrans)))
	{
		if( RC_OK( rc = fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				uiIxNum, NULL, &pIxd)))
		{
	
			/* Build the collated key */

			rc = KYTreeToKey( pDb, pIxd, pRecord, uiContainer,
					pKeyBuf, puiKeyLenRV, uiFlag );
		}
	}

//Exit:
	if( bImplicitTrans)
		(void)flmAbortDbTrans( pDb);
	(void)fdbExit( pDb);
	return( rc);
}
