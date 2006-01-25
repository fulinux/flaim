//-------------------------------------------------------------------------
// Desc:	Dictionary access routiones.
// Tabs:	3
//
//		Copyright (c) 1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fdict.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/***************************************************************************
Desc:		Get the field information.  Try shared first and then local.
****************************************************************************/
RCODE fdictGetField(
	FDICT *		pDict,
	FLMUINT		uiFieldNum,					// [in] Field Number to look up
	FLMUINT *	puiFieldType, 				// [out] Optional
	IFD **		ppFirstIfd,					// [out] Optional
	FLMUINT *	puiFieldState)				// [out] Optional
{
	RCODE			rc = FERR_OK;
	ITT			nonstandardItt;
	ITT *			pItt;

	if( pDict && pDict->pIttTbl && uiFieldNum < pDict->uiIttCnt)
	{
		pItt = &pDict->pIttTbl[ uiFieldNum];
		
		// Is it really a field?

		if( ! ITT_IS_FIELD( pItt))
		{
			rc = RC_SET( FERR_BAD_FIELD_NUM);
			goto Exit;
		}
	}
	else
	{
		// Check if the field is a FLAIM dictionary field.
		// Most of these fields are TEXT fields.

		if( (uiFieldNum >= FLM_DICT_FIELD_NUMS)
		&&  (uiFieldNum <= FLM_LAST_DICT_FIELD_NUM))
		{
			// Most of the dictionary fields are text type.
			// KYBUILD now doesn't verify unregistered or dictionary fields types.

			pItt = &nonstandardItt;
			nonstandardItt.uiType = FLM_TEXT_TYPE;
			nonstandardItt.pvItem = NULL;
		}
		else if( uiFieldNum >= FLM_UNREGISTERED_TAGS)
		{
			pItt = &nonstandardItt;
			nonstandardItt.uiType = FLM_TEXT_TYPE;
			nonstandardItt.pvItem = NULL;
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_NUM);
			goto Exit;
		}

	}
	if( puiFieldType)
	{
		*puiFieldType = ITT_FLD_GET_TYPE( pItt);
	}
	if( ppFirstIfd)
	{
		*ppFirstIfd = (IFD *)pItt->pvItem;
	}
	if( puiFieldState)
	{
		*puiFieldState = ITT_FLD_GET_STATE( pItt);
	}

Exit:
	return( rc);
}

/***************************************************************************
Desc:		Get the encryption information.
****************************************************************************/
RCODE fdictGetEncInfo(
	FDB *			pDb,
	FLMUINT		uiEncId,					// [in] Encryption definition to look up
	FLMUINT *	puiEncType,				// [out] Optional
	FLMUINT *	puiEncState				// [out] Optional
	)
{
	RCODE			rc = FERR_OK;
	ITT *			pItt;
	FDICT *		pDict = pDb->pDict;
	FlmRecord *	pRecord = NULL;
	void *		pvField = NULL;
	FLMUINT		uiEncState;
	FLMUINT		uiEncType;
	
	if ( pDb->pFile->bInLimitedMode)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if( pDict && pDict->pIttTbl && uiEncId < pDict->uiIttCnt)
	{
		pItt = &pDict->pIttTbl[ uiEncId];

		// Is it really an encryption definition?

		if( ! ITT_IS_ENCDEF( pItt))
		{
			rc = RC_SET( FERR_BAD_ENCDEF_ID);
			goto Exit;
		}

		uiEncType = ((F_CCS *)pItt->pvItem)->getEncType();

		// Get the Encryption record and determine the state.
		if (RC_BAD( rc = FlmRecordRetrieve(	(HFDB)pDb,
														FLM_DICT_CONTAINER,
														uiEncId,
														FO_EXACT,
														&pRecord,
														NULL)))
		{
			goto Exit;
		}

		pvField = pRecord->find( pRecord->root(),
										 FLM_STATE_TAG);
		if (pvField)
		{
			const char *	pDataPtr = (const char *)pRecord->getDataPtr( pvField);

			if (f_strnicmp( pDataPtr, "chec", 4) == 0)
			{
				uiEncState = ITT_ENC_STATE_CHECKING;
			}
			else if (f_strnicmp( pDataPtr, "purg", 4) == 0)
			{
				uiEncState = ITT_ENC_STATE_PURGE;
			}
			else if (f_strnicmp( pDataPtr, "acti", 4) == 0)
			{
				uiEncState = ITT_ENC_STATE_ACTIVE;
			}
			else
			{
				uiEncState = ITT_ENC_STATE_UNUSED;
			}
		}
		else
		{
			uiEncState = ITT_ENC_STATE_UNUSED;
		}
	}
	else
	{
		rc = RC_SET( FERR_BAD_ENCDEF_ID);
		goto Exit;
	}

	if( puiEncType)
	{
		*puiEncType = uiEncType;
	}
	if( puiEncState)
	{
		*puiEncState = uiEncState;
	}

Exit:

	if (pRecord)
	{
		pRecord->Release();
	}
	return( rc);
}

/***************************************************************************
Desc:		Get the Container given a container number.
****************************************************************************/
RCODE fdictGetContainer(
	FDICT *		pDict,
	FLMUINT		uiContNum,
	LFILE **		ppLFile)
{
	ITT *			pItt;

	if( pDict && uiContNum < pDict->uiIttCnt && pDict->pIttTbl)
	{
		pItt = &pDict->pIttTbl[ uiContNum];
		
		// Is it really a container?

		if( !ITT_IS_CONTAINER( pItt))
		{
			return( RC_SET( FERR_BAD_CONTAINER));
		}
		if( ppLFile)
		{
			*ppLFile = (LFILE *) pItt->pvItem;
		}
	}
	else
	{
		// Hard coded container - data is [0], dictionary is [1].

		if( uiContNum == FLM_DATA_CONTAINER)
		{
			if( ppLFile)
			{
				*ppLFile = &pDict->pLFileTbl[ LFILE_DATA_CONTAINER_OFFSET];
			}
		}
		else if( uiContNum == FLM_DICT_CONTAINER)
		{
			if( ppLFile)
			{
				*ppLFile = &pDict->pLFileTbl[ LFILE_DICT_CONTAINER_OFFSET];
			}
		}
		else if( uiContNum == FLM_TRACKER_CONTAINER)
		{
			if( ppLFile)
			{
				*ppLFile = &pDict->pLFileTbl[ LFILE_TRACKER_CONTAINER_OFFSET];
			}
		}
		else
		{
			return( RC_SET( FERR_BAD_CONTAINER));
		}
	}

	return( FERR_OK);
}

/***************************************************************************
Desc:		Get the IXD, LFILE and IFD information given an index number.
****************************************************************************/
RCODE fdictGetIndex(
	FDICT *		pDict,
	FLMBOOL		bInLimitedMode,
	FLMUINT		uiIxNum,
	LFILE **		ppLFile,		// [out] optional
	IXD **		ppIxd,		// [out] optional
	FLMBOOL		bOfflineOk)
{
	RCODE			rc = FERR_OK;
	ITT *			pItt;
	LFILE *		pLFile;
	IXD *			pIxd;

	if( ppIxd)
	{
		*ppIxd = NULL;
	}

	if( ppLFile)
	{
		*ppLFile = NULL;
	}

	if( pDict && uiIxNum < pDict->uiIttCnt && pDict->pIttTbl)
	{
		pItt = &pDict->pIttTbl[ uiIxNum];

		// Is it really a container?

		if( !ITT_IS_INDEX( pItt))
		{
			rc = RC_SET( FERR_BAD_IX);
			goto Exit;
		}
		pLFile = (LFILE *) pItt->pvItem;
		pIxd = pLFile->pIxd;

		if( ppLFile)
		{
			*ppLFile = pLFile;
		}

		if( ppIxd)
		{
			*ppIxd = pIxd;
		}

		// If the index is suspended the IXD_OFFLINE flag
		// will be set, so it is sufficient to just test
		// the IXD_OFFLINE for both suspended and offline
		// conditions.

		if( (pIxd->uiFlags & IXD_OFFLINE) && !bOfflineOk)
		{
			rc = RC_SET( FERR_INDEX_OFFLINE);
			goto Exit;
		}

		// An encrypted index that cannot be decrypted is as good as
		// offline.
		if ( pIxd->uiEncId && bInLimitedMode && !bOfflineOk)
		{
			rc = RC_SET( FERR_INDEX_OFFLINE);
			goto Exit;
		}
	}
	else if (uiIxNum == FLM_DICT_INDEX)
	{
		pLFile = pDict->pLFileTbl + LFILE_DICT_INDEX_OFFSET;
		if( ppLFile)
		{
			*ppLFile = pLFile;
		}
		if( ppIxd)
		{
			*ppIxd = pLFile->pIxd;
		}
	}
	else
	{
		rc = RC_SET( FERR_BAD_IX);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Given a IXD ID (index drn), returns the next Index Def (IXD).
****************************************************************************/
RCODE fdictGetNextIXD(
	FDICT *		pDict,
	FLMUINT		uiIndexNum,
	IXD **		ppIxd)
{
	RCODE			rc = FERR_OK;
	IXD *			pIxd = NULL;

	flmAssert( pDict && pDict->uiIxdCnt);

	for( uiIndexNum++; uiIndexNum < pDict->uiIttCnt; uiIndexNum++)
	{
		ITT * pItt = &pDict->pIttTbl[ uiIndexNum];
		if( ITT_IS_INDEX( pItt))
		{
			LFILE * pLFile = (LFILE *) pItt->pvItem;
			pIxd = pLFile->pIxd;
			break;
		}
	}

	if( !pIxd && uiIndexNum < FLM_DICT_INDEX)
	{
		// Special case -- return the dictionary index
		pIxd = pDict->pIxdTbl;
	}

	if( pIxd)
	{
		// Check to see if the index is offline.  Still return *ppIxd.

		if( pIxd->uiFlags & IXD_OFFLINE)
		{
			rc = RC_SET( FERR_INDEX_OFFLINE);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

Exit:

	// Always set the return parm.

	if( ppIxd)
	{
		*ppIxd = pIxd;
	}

	return( rc);
}
