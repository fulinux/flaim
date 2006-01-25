//-------------------------------------------------------------------------
// Desc:	Export/Import support functions.
// Tabs:	3
//
//		Copyright (c) 1995-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fexpimp.cpp 12319 2006-01-19 15:52:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define BINARY_GED_HEADER_LEN	8

static FLMBYTE FlmBinaryGedHeader [BINARY_GED_HEADER_LEN]
	= { 0xFF, 'F', 'L', 'M', 'D', 'I', 'C', 'T' };

static FLMBYTE FlmBinaryRecHeader [BINARY_GED_HEADER_LEN]
	= { 0xFF, 'F', 'L', 'M', 'R', 'E', 'C', 'S' };

/* Local function prototypes. */

FSTATIC RCODE expWrite(
	EXP_IMP_INFO_p			pExpImpInfo,
	const FLMBYTE *		pData,
	FLMUINT					uiDataLen);

FSTATIC RCODE impRead(
	EXP_IMP_INFO_p			pExpImpInfo,
	FLMBYTE *				pData,
	FLMUINT					uiDataLen,
	FLMUINT *				puiBytesReadRV);

/****************************************************************************
Desc:		Initializes the export/import information for reading or writing.
****************************************************************************/
RCODE expImpInit(
	F_FileHdl *		pFileHdl,		/* File we are going to be exporting to or
												importing from. */
	FLMUINT			uiFlag,			/* Flag indicating whether we are exporting
												or importing and what kind of data.
												EXPIMP_IMPORT_DICTIONARY
												EXPIMP_EXPORT_DICTIONARY
												EXPIMP_IMPORT_EXPORT_GEDCOM */
	EXP_IMP_INFO_p	pExpImpInfoRV	/* Export/Import info. structure that is
												to be initialized. */
	)
{
	RCODE		rc;

	f_memset( pExpImpInfoRV, 0, sizeof( EXP_IMP_INFO));
	pExpImpInfoRV->pFileHdl = pFileHdl;
	pExpImpInfoRV->bDictRecords = (uiFlag == EXPIMP_IMPORT_EXPORT_GEDCOM)
												? FALSE : TRUE;
	
	/* Allocate a buffer for reading or writing. */

	pExpImpInfoRV->uiBufSize = (uiFlag == EXPIMP_IMPORT_EXPORT_GEDCOM) 
										? (FLMUINT) 2048 : (FLMUINT) 32768;
	for (;;)
	{
		if( RC_BAD( rc = f_alloc( 
			pExpImpInfoRV->uiBufSize, &pExpImpInfoRV->pBuf)))
		{
			pExpImpInfoRV->uiBufSize -= 512;
			if (pExpImpInfoRV->uiBufSize < 1024)
			{
				pExpImpInfoRV->uiBufSize = 0;
				goto Exit;
			}
		}
		else
			break;
	}

	/* If writing, output the header data.  If reading, seek past it. */

	if( uiFlag == EXPIMP_EXPORT_DICTIONARY)
	{

		/* Write out the header data. */

		rc = expWrite( pExpImpInfoRV, FlmBinaryGedHeader,
							BINARY_GED_HEADER_LEN);
	}
	else if( uiFlag == EXPIMP_IMPORT_DICTIONARY)
	{
		rc = pFileHdl->Seek( (FLMUINT)BINARY_GED_HEADER_LEN,
							F_IO_SEEK_SET, &pExpImpInfoRV->uiFilePos);
	}
	else
	{
		rc = expWrite( pExpImpInfoRV, FlmBinaryRecHeader,
							BINARY_GED_HEADER_LEN);
	}	
Exit:
	if (RC_BAD( rc))
		expImpFree( pExpImpInfoRV);
	return( rc);
}

/****************************************************************************
Desc:	Frees up the buffers used to do reading/writing during an export or
		import.
****************************************************************************/
void expImpFree(
	EXP_IMP_INFO_p	pExpImpInfo		/* Export/Import information. */
	)
{
	if (pExpImpInfo->pBuf)
		f_free( &pExpImpInfo->pBuf);
	f_memset( pExpImpInfo, 0, sizeof( EXP_IMP_INFO));
}

/****************************************************************************
Desc:	Flush the current export buffer to disk.
****************************************************************************/
RCODE expFlush(
	EXP_IMP_INFO_p	pExpImpInfo	)	/* Export/Import information. */
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiBytesWritten;

	if( (pExpImpInfo->uiBufUsed) && (pExpImpInfo->bBufDirty))
	{
		if( RC_BAD( rc = pExpImpInfo->pFileHdl->Write(
				pExpImpInfo->uiFilePos,
				pExpImpInfo->uiBufUsed, pExpImpInfo->pBuf, &uiBytesWritten)))
			goto Exit;
		if( uiBytesWritten < pExpImpInfo->uiBufUsed)
		{
			rc = RC_SET( FERR_IO_DISK_FULL);
			goto Exit;
		}
		pExpImpInfo->uiFilePos += uiBytesWritten;
		pExpImpInfo->uiCurrBuffOffset =
		pExpImpInfo->uiBufUsed = 0;
		pExpImpInfo->bBufDirty = FALSE;
 	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Seek to an absolute offset in the export/import file.
****************************************************************************/
RCODE expImpSeek(
	EXP_IMP_INFO_p	pExpImpInfo,	/* Export/Import information. */
	FLMUINT			uiSeekPos		/* Absolute offset to seek to. */
	)
{
	RCODE	rc = FERR_OK;

	if ((uiSeekPos >= pExpImpInfo->uiFilePos) &&
		 (uiSeekPos < pExpImpInfo->uiFilePos + (FLMUINT)pExpImpInfo->uiBufUsed))
	{
		pExpImpInfo->uiCurrBuffOffset =
				(FLMUINT)(uiSeekPos - pExpImpInfo->uiFilePos);
	}
	else
	{
		if (pExpImpInfo->bBufDirty)
		{
			if (RC_BAD( rc = expFlush( pExpImpInfo)))
				goto Exit;
		}
		pExpImpInfo->uiFilePos = uiSeekPos;
		pExpImpInfo->uiBufUsed = pExpImpInfo->uiCurrBuffOffset = 0;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Writes data to the export file via the export buffer.
****************************************************************************/
FSTATIC RCODE expWrite(
	EXP_IMP_INFO_p			pExpImpInfo,
	const FLMBYTE *		pData,
	FLMUINT					uiDataLen)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCopyLen;

	while( uiDataLen)
	{
		if ((uiCopyLen =
			  (pExpImpInfo->uiBufSize - pExpImpInfo->uiCurrBuffOffset)) > uiDataLen)
			uiCopyLen = uiDataLen;
		f_memcpy( &pExpImpInfo->pBuf [pExpImpInfo->uiCurrBuffOffset],
					 pData, uiCopyLen);
		pExpImpInfo->bBufDirty = TRUE;
		uiDataLen -= uiCopyLen;
		pData += uiCopyLen;
		pExpImpInfo->uiCurrBuffOffset += uiCopyLen;
		if (pExpImpInfo->uiCurrBuffOffset > pExpImpInfo->uiBufUsed)
			pExpImpInfo->uiBufUsed = pExpImpInfo->uiCurrBuffOffset;
		if (pExpImpInfo->uiCurrBuffOffset == pExpImpInfo->uiBufSize)
		{
			if (RC_BAD( rc = expFlush( pExpImpInfo)))
				goto Exit;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Writes one FLAIM record to a binary GEDCOM file.
****************************************************************************/
RCODE expWriteRec(
	EXP_IMP_INFO_p	pExpImpInfo,	/* Buffer info. for export file. */
	FlmRecord *		pRecord,			/* record to be written out. */
	FLMUINT			uiDrn)			/* DRN of GEDCOM record being written out. */
{
	RCODE			rc = FERR_OK;
	FLMBYTE		TBuf [24];
	FLMUINT		uiLen;
	FLMUINT		uiTagNum;
	FLMUINT		uiInitLevel;
	FLMBOOL		bOutputtingRecInfo;
	FLMBOOL		bRootNode;
	FLMUINT		uiTmpLen;
	FlmRecord * pRec = NULL;
	FlmRecord *	pRecInfoRec = NULL;
	void *		pvField;
	
	if( pExpImpInfo->bDictRecords)
	{
		// Create a record for the RECINFO information

		if( (pRecInfoRec = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pRecInfoRec->insertLast( 0, 
			FLM_RECINFO_TAG, FLM_NUMBER_TYPE, &pvField)))
		{
			goto Exit;
		}

		/* Add the record's DRN to the RECINFO information. */
	
		if( RC_BAD( rc = flmAddField( pRecInfoRec, FLM_DRN_TAG,
								(void *)&uiDrn, 4, FLM_NUMBER_TYPE)))
			goto Exit;

		bOutputtingRecInfo = TRUE;

		/* Output both the REC_INFO GEDCOM tree and the record's GEDCOM tree. */

		bRootNode = FALSE;
		pRec = pRecInfoRec;
	}
	else
	{
		/* Output only the GEDCOM tree. */
		
		bOutputtingRecInfo = FALSE;
		bRootNode = TRUE;
		pRec = pRecord;
	}


	for(;;)
	{
		/* Output each node in the record. */

		pvField = pRec->root();
		uiInitLevel = pRec->getLevel( pvField);
		do
		{
			uiTagNum = pRec->getFieldID( pvField);
			uiLen		= pRec->getDataLength( pvField);
			UW2FBA( (FLMUINT16)uiTagNum, TBuf);
			UW2FBA( (FLMUINT16)uiLen, &TBuf[ 2]);
			TBuf[ 4] = (FLMBYTE)( pRec->getLevel( pvField) - uiInitLevel);
			TBuf[ 5] = (FLMBYTE)( pRec->getDataType( pvField));

			/* Add on the record source information for the root node. */

			uiTmpLen = 6;
			if( bRootNode)
			{
				// UD2FBA( pRec->getDatabaseID(), &TBuf[ 6]); <== hDb not supported
				UW2FBA( (FLMUINT16)pRec->getContainerID(), &TBuf[ 14]);
				UD2FBA( pRec->getID(), &TBuf[ 16]);
				uiTmpLen = 20;
											
				bRootNode = FALSE;
			}

			if( RC_BAD( rc = expWrite( pExpImpInfo, TBuf, uiTmpLen)))
				goto Exit;

			if( uiLen)
			{
				const FLMBYTE *	pvData = pRec->getDataPtr( pvField);

				if( RC_BAD( rc = expWrite( pExpImpInfo, pvData, uiLen))) 
				{
					goto Exit;
				}
			}

			pvField = pRec->next( pvField);

		} while( pvField && (pRec->getLevel( pvField) > uiInitLevel));

		/* Output a zero tag number to indicate end of GEDCOM record. */

		UW2FBA( 0, TBuf);
		if( RC_BAD( rc = expWrite( pExpImpInfo, TBuf, 2)))
			goto Exit;

		/* Set things up to output the record after the REC_INFO. */

		if( !bOutputtingRecInfo)
			break;
		bOutputtingRecInfo = FALSE;
		bRootNode = TRUE;
		pRec = pRecord;
	}

Exit:

	if( pRecInfoRec)
	{
		pRecInfoRec->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Reads data from the import file via the import buffer.
****************************************************************************/
FSTATIC RCODE impRead(
	EXP_IMP_INFO_p	pExpImpInfo,	/* Export/Import information. */
	FLMBYTE *		pData,			/* Buffer where data is to be read into. */
	FLMUINT			uiDataLen,		/* Length of data to be read in. */
	FLMUINT *		puiBytesReadRV)	/* Returns amount of data read in. */
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiCopyLen;
	FLMUINT	uiBytesRead = 0;

	while (uiDataLen)
	{

		/* See if we need to read some more data into the import buffer. */

		if (pExpImpInfo->uiCurrBuffOffset == pExpImpInfo->uiBufUsed)
		{

			/* If we have a dirty buffer, flush it out first. */

			if (pExpImpInfo->bBufDirty)
			{
				if (RC_BAD( rc = expFlush( pExpImpInfo)))
					goto Exit;
			}
			else
			{
				pExpImpInfo->uiFilePos += (FLMUINT)pExpImpInfo->uiBufUsed;
				pExpImpInfo->uiBufUsed = pExpImpInfo->uiCurrBuffOffset = 0;
			}
			if (RC_BAD( rc = pExpImpInfo->pFileHdl->Read(
												pExpImpInfo->uiFilePos,
												pExpImpInfo->uiBufSize,
												pExpImpInfo->pBuf,
												&pExpImpInfo->uiBufUsed)))
			{
				if ((rc == FERR_IO_END_OF_FILE) && (pExpImpInfo->uiBufUsed))
					rc = FERR_OK;
				else
					goto Exit;
			}
		}

		/* Copy from the import buffer to the data buffer. */

		if ((uiCopyLen =
			  (pExpImpInfo->uiBufUsed - pExpImpInfo->uiCurrBuffOffset)) > uiDataLen)
			uiCopyLen = uiDataLen;
		f_memcpy( pData, &pExpImpInfo->pBuf [pExpImpInfo->uiCurrBuffOffset],
					 uiCopyLen);	
		uiDataLen -= uiCopyLen;
		uiBytesRead += uiCopyLen;
		pData += uiCopyLen;
		pExpImpInfo->uiCurrBuffOffset += uiCopyLen;
	}
Exit:
	*puiBytesReadRV = uiBytesRead;
	return( rc);
}

/****************************************************************************
Desc:	Reads one GEDCOM record from an export/import file.
****************************************************************************/
RCODE impReadRec(
	EXP_IMP_INFO_p	pExpImpInfo,	/* Export/Import information. */
	FlmRecord **	ppRecordRV)		/* Returns record that was read in. */
{
	RCODE			rc = FERR_OK;
	FLMBYTE		TBuf [24];
	FLMUINT		uiLen;
	FLMUINT		uiTagNum;
	FLMUINT		uiRecInfoDrn = 0;
	FLMUINT		uiDictID;
	FLMBOOL		bHaveRecInfo = FALSE;
	FLMBOOL		bHaveDictID = FALSE;
	FLMUINT		uiLevel;
	FLMUINT		uiType;
	FLMBOOL		bGettingRecInfo;
	FLMUINT		uiBytesRead;
	FLMUINT		uiTmpLen;
	FlmRecord *	pRecord = NULL;
	void *		pvField;

	bGettingRecInfo = (pExpImpInfo->bDictRecords) ? TRUE : FALSE;

	/* Read each node in the REC_INFO (if dictionary) and then the record. */

	for (;;)
	{
		if (RC_BAD( rc = impRead( pExpImpInfo, TBuf, 2, &uiBytesRead)))
		{
			if ((rc == FERR_IO_END_OF_FILE) && (uiBytesRead == 0) &&
				 ((!bGettingRecInfo) || (!bHaveRecInfo)))
			{
				rc = RC_SET( FERR_END);
			}
			goto Exit;
		}

		/* A tag number of zero means we are at the end of the record. */

		uiTagNum = FB2UW( TBuf);
		if (!uiTagNum)
		{
			if (bGettingRecInfo)
			{
				bGettingRecInfo = FALSE;
				continue;
			}
			else
				break;
		}
		uiTmpLen = ((!bGettingRecInfo) && (!pRecord))
						? 18
						: 4;
		if (RC_BAD( rc = impRead( pExpImpInfo, TBuf, uiTmpLen, &uiBytesRead)))
			goto Exit;
		uiLen = FB2UW( TBuf);
		uiLevel = TBuf [2];
		uiType = TBuf [3];

		if( !pRecord)
		{
			if( (pRecord = f_new FlmRecord) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			pRecord->setContainerID( FB2UW( &TBuf[ 12]));
			pRecord->setID( FB2UD( &TBuf[ 14]));
		}

		if( RC_BAD( rc = pRecord->insertLast( uiLevel, uiTagNum, uiType, &pvField)))
		{
			goto Exit;
		}

		if( uiLen)
		{		
			FLMBYTE * pValue;
			
			if (RC_BAD( rc = pRecord->allocStorageSpace( pvField, uiType,
														uiLen, 0, 0, 0, &pValue, NULL)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = impRead( pExpImpInfo, pValue, uiLen, &uiBytesRead)))
				goto Exit;
		}

		/* Link the node into the tree. */

		if( bGettingRecInfo)
		{
			switch( uiTagNum)
			{
				case FLM_RECINFO_TAG:
					bHaveRecInfo = TRUE;
					break;
				case FLM_DRN_TAG:
					if( RC_BAD( rc = pRecord->getUINT( pvField, &uiRecInfoDrn)))
						goto Exit;
					break;
				case FLM_DICT_SEQ_TAG:
					if( RC_BAD( rc = pRecord->getUINT( pvField, &uiDictID)))
						goto Exit;
					bHaveDictID = TRUE;
					break;
			}
		}
	}

Exit:
	if( RC_OK( rc))
	{
		*ppRecordRV = pRecord;
	}
	else
	{
		if( pRecord)
		{
			pRecord->Release();
		}

		*ppRecordRV = NULL;
	}
	return( rc);
}

/****************************************************************************
Desc:	Tests to see if a file is a binary export/import file.  After the
		call is over, we return the file position to whatever location
		it was at before this call was made.
****************************************************************************/
RCODE impFileIsExpImp(
	F_FileHdl *	pFileHdl,				/* Open file handle. */
	FLMBOOL *	pbFileIsBinaryRV)		/* Returns TRUE or FALSE to indicate if
													file is a binary GEDCOM file. */
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiCurrPos;
	FLMUINT	uiIgnore;
	FLMBYTE	byHeader [BINARY_GED_HEADER_LEN];
	FLMUINT	uiBytesRead;

	*pbFileIsBinaryRV = FALSE;

	/* Save current position so we can return to it. */

	if (RC_BAD( rc = pFileHdl->Seek( (FLMUINT)0, F_IO_SEEK_CUR, &uiCurrPos)))
		goto Exit;

	/* Read the file's header information. */

	if (RC_BAD( rc = pFileHdl->Read( (FLMUINT)0, BINARY_GED_HEADER_LEN,
									byHeader, &uiBytesRead)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			uiBytesRead = 0;
			rc = FERR_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if( (uiBytesRead == BINARY_GED_HEADER_LEN) &&
		 ((f_memcmp( byHeader, FlmBinaryGedHeader, BINARY_GED_HEADER_LEN) == 0) ||
		  (f_memcmp( byHeader, FlmBinaryRecHeader, BINARY_GED_HEADER_LEN) == 0)))
	{
		*pbFileIsBinaryRV = TRUE;
	}

	/* Reset the file position to where it was before. */

	rc = pFileHdl->Seek( uiCurrPos, F_IO_SEEK_SET, &uiIgnore);

Exit:
	return( rc);
}


