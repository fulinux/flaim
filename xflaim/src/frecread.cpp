//------------------------------------------------------------------------------
// Desc:	Routines for reading records from FLXIM 4.x databases
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frecread.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

static FLMBYTE gv_ucMaxBcdINT32[] = {0x21, 0x47, 0x48, 0x36, 0x47};
static FLMBYTE gv_ucMinBcdINT32[] = {0xB2, 0x14, 0x74, 0x83, 0x64, 0x8F};
static FLMBYTE gv_ucMaxBcdUINT32[] = {0x42, 0x94, 0x96, 0x72, 0x95};

typedef struct
{
	const char *	pszTagName;
	FLMUINT			uiTagNum;
	FLMUINT			uiFieldType;
} FLM_4x_DICT_TAG_INFO;

FLM_4x_DICT_TAG_INFO Flm4xDictTagInfo[] =
{
	{FLM_4x_FIELD_TAG_NAME, FLM_4x_FIELD_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_INDEX_TAG_NAME, FLM_4x_INDEX_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_TYPE_TAG_NAME, FLM_4x_TYPE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_CONTAINER_TAG_NAME, FLM_4x_CONTAINER_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_LANGUAGE_TAG_NAME, FLM_4x_LANGUAGE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_OPTIONAL_TAG_NAME, FLM_4x_OPTIONAL_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_UNIQUE_TAG_NAME, FLM_4x_UNIQUE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_KEY_TAG_NAME, FLM_4x_KEY_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_REFS_TAG_NAME, FLM_4x_REFS_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_AREA_TAG_NAME, FLM_4x_AREA_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_STATE_TAG_NAME, FLM_4x_STATE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_BLOB_TAG_NAME, FLM_4x_BLOB_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_THRESHOLD_TAG_NAME, FLM_4x_THRESHOLD_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_SUFFIX_TAG_NAME, FLM_4x_SUFFIX_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_SUBDIRECTORY_TAG_NAME, FLM_4x_SUBDIRECTORY_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_RESERVED_TAG_NAME, FLM_4x_RESERVED_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_SUBNAME_TAG_NAME, FLM_4x_SUBNAME_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_NAME_TAG_NAME, FLM_4x_NAME_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_BASE_TAG_NAME, FLM_4x_BASE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_CASE_TAG_NAME, FLM_4x_CASE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_COMBINATIONS_TAG_NAME, FLM_4x_COMBINATIONS_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_COUNT_TAG_NAME, FLM_4x_COUNT_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_POSITIONING_TAG_NAME, FLM_4x_POSITIONING_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_PAIRED_TAG_NAME, FLM_4x_PAIRED_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_PARENT_TAG_NAME, FLM_4x_PARENT_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_POST_TAG_NAME, FLM_4x_POST_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_REQUIRED_TAG_NAME, FLM_4x_REQUIRED_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_USE_TAG_NAME, FLM_4x_USE_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_FILTER_TAG_NAME, FLM_4x_FILTER_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_LIMIT_TAG_NAME, FLM_4x_LIMIT_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_DICT_TAG_NAME, FLM_4x_DICT_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_RECINFO_TAG_NAME, FLM_4x_RECINFO_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_DRN_TAG_NAME, FLM_4x_DRN_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_DICT_SEQ_TAG_NAME, FLM_4x_DICT_SEQ_TAG, FLM_4x_TEXT_TYPE},
	{FLM_4x_LAST_CONTAINER_INDEXED_TAG_NAME, FLM_4x_LAST_CONTAINER_INDEXED_TAG, FLM_4x_NUMBER_TYPE},
	{FLM_4x_LAST_DRN_INDEXED_TAG_NAME, FLM_4x_LAST_DRN_INDEXED_TAG, FLM_4x_NUMBER_TYPE},
	{FLM_4x_ONLINE_TRANS_ID_TAG_NAME, FLM_4x_ONLINE_TRANS_ID_TAG, FLM_4x_NUMBER_TYPE},
	{NULL, 0}
};

/***************************************************************************
Desc:		
****************************************************************************/
F_4xReader::F_4xReader()
{
	m_tmpPool.poolInit( 32 * 1024);
	m_pSuperHdl = NULL;
	m_pLckFile = NULL;
	m_uiMaxFileSize = 0;
	m_pLFileTbl = NULL;
	m_uiLFileCnt = 0;
	m_uiFieldTblSize = 0;
	m_puiFieldTbl = NULL;
	m_ppBlockTbl = NULL;
	m_uiBlockTblSize = 0;
	m_uiDefaultContainer = 0;
	m_pNameTable = NULL;
	f_memset( &m_fileHdr, 0, sizeof( F_4x_FILE_HDR));
	f_memset( &m_logHdr, 0, sizeof( F_4x_LOG_HDR));
}

/***************************************************************************
Desc:		
****************************************************************************/
F_4xReader::~F_4xReader()
{
	closeDatabase();
	m_tmpPool.poolFree();
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::openDatabase(
	char *		pszPath)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucReadBuf = NULL;
	FLMBYTE *		pucPrefix;
	FLMBYTE *		pucFileHdr;
	FLMBYTE *		pucLogHdr;
	FLMUINT			uiBytesRead;
	FLMUINT			uiTmp;
	F_FileHdl *		pFileHdl = NULL;

	flmAssert( !m_pSuperHdl);

	if( (m_pSuperHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pSuperHdl->Setup( NULL, pszPath, NULL)))
	{
		goto Exit;
	}

	// We must have exclusive access.

	if( RC_BAD( rc = createLckFile( pszPath)))
	{
		goto Exit;
	}

	// Read and verify the file and log headers. 

	if( RC_BAD( rc = m_pSuperHdl->GetFileHdl( 0, FALSE, 
		(IF_FileHdl **)&pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 2048, &pucReadBuf)))
	{
		goto Exit;
	}

	// Read the fixed information area

	if( RC_BAD( rc = pFileHdl->Read( 0, 2048, pucReadBuf, &uiBytesRead)))
	{
		goto Exit;
	}
	
	*pucReadBuf = 0xFF;
	pucPrefix = pucReadBuf;
	pucFileHdr = &pucReadBuf[ FLM_4x_FLAIM_HEADER_START];

	// Make sure we have a valid prefix

	if( pucPrefix[ 1] != f_toascii('W') ||
		 pucPrefix[ 2] != f_toascii('P') ||
		 pucPrefix[ 3] != f_toascii('C'))
	{
		rc = RC_SET( NE_XFLM_NOT_FLAIM);
		goto Exit;
	}

	// Extract the file header info

	m_fileHdr.uiBlockSize = (FLMUINT)FB2UW( &pucFileHdr[ FLM_4x_DB_BLOCK_SIZE]);
	m_fileHdr.uiAppMajorVer = pucPrefix[ 10];
	m_fileHdr.uiAppMinorVer = pucPrefix[ 11];
	m_fileHdr.uiDefaultLanguage = pucFileHdr[ FLM_4x_DB_DEFAULT_LANGUAGE];
	m_fileHdr.uiVersionNum = 
		((FLMUINT16)(pucFileHdr[ FLM_4x_VER_POS] - ASCII_ZERO) * 100 +
			(FLMUINT16)(pucFileHdr[ FLM_4x_MINOR_VER_POS] - ASCII_ZERO) * 10 +
			(FLMUINT16)(pucFileHdr[ FLM_4x_SMINOR_VER_POS] - ASCII_ZERO));

	// Is the block size valid?

	if( m_fileHdr.uiBlockSize != 4096 && 
		m_fileHdr.uiBlockSize != 8192)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Supported version?

	switch( m_fileHdr.uiVersionNum)
	{
		case FLM_VER_4_0:
		case FLM_VER_4_3:
		case FLM_VER_4_31:
		case FLM_VER_4_50:
		case FLM_VER_4_51:
			break;
		default:
			rc = RC_SET( NE_XFLM_UNSUPPORTED_VERSION);
			goto Exit;
	}

	// Get other log header elements.

	m_fileHdr.uiFirstLFHBlkAddr =
		(FLMUINT)FB2UD( &pucFileHdr[ FLM_4x_DB_1ST_LFH_ADDR]);

	if( pucFileHdr[ FLM_4x_FLAIM_NAME_POS     ] != f_toascii( 'F') ||
		 pucFileHdr[ FLM_4x_FLAIM_NAME_POS + 1 ] != f_toascii( 'L') || 
		 pucFileHdr[ FLM_4x_FLAIM_NAME_POS + 2 ] != f_toascii( 'A') || 
		 pucFileHdr[ FLM_4x_FLAIM_NAME_POS + 3 ] != f_toascii( 'I') || 
		 pucFileHdr[ FLM_4x_FLAIM_NAME_POS + 4 ] != f_toascii( 'M'))
	{
		rc = RC_SET( NE_XFLM_NOT_FLAIM);
		goto Exit;
	}

	// Set up the uiSigBitsInBlkSize member of the file
	// header

	m_fileHdr.uiSigBitsInBlkSize = 0;
	uiTmp = m_fileHdr.uiBlockSize;
	while( !(uiTmp & 0x0001))
	{
		m_fileHdr.uiSigBitsInBlkSize++;
		uiTmp >>= 1;
	}

	// Get the log file header information

	pucLogHdr = &pucReadBuf[ FLM_4x_DB_LOG_HEADER_START];

	// Verify the checksums in the log header

	if( lgHdrCheckSum( pucLogHdr, TRUE) != 0)
	{
		rc = RC_SET( NE_XFLM_BLOCK_CRC);
		goto Exit;
	}

	m_logHdr.uiCurrTransID =
		(FLMUINT)FB2UD( &pucLogHdr[ FLM_4x_LOG_CURR_TRANS_ID]);

	m_logHdr.uiLogicalEOF =
		(FLMUINT)FB2UD( &pucLogHdr[ FLM_4x_LOG_LOGICAL_EOF]);

	m_logHdr.uiFirstAvailBlkAddr =
		(FLMUINT)FB2UD( &pucLogHdr[ FLM_4x_LOG_PF_AVAIL_BLKS]);

	m_logHdr.uiAvailBlkCount =
		(FLMUINT)FB2UD( &pucLogHdr[ FLM_4x_LOG_PF_NUM_AVAIL_BLKS]);

	// Get the maximum file size

	if( m_fileHdr.uiVersionNum >= FLM_VER_4_3)
	{
		m_uiMaxFileSize = (FLMUINT)(FB2UW(&((pucLogHdr)[ 
			FLM_4x_LOG_MAX_FILE_SIZE]))) << 16;
	}
	else
	{
		m_uiMaxFileSize = 0x7FF00000;
	}
	
	// Make sure that no recovery needs to be done.

	if( (FLMUINT)FB2UD( &pucLogHdr[ FLM_4x_LOG_ROLLBACK_EOF]) != 
			m_fileHdr.uiBlockSize ||
		(FLMUINT)FB2UD( &pucLogHdr[ FLM_4x_LOG_PL_FIRST_CP_BLOCK_ADDR]))
	{
		rc = RC_SET( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	// Set the block size

	m_pSuperHdl->ReleaseFile( (FLMUINT)0, TRUE);
	m_pSuperHdl->SetBlockSize( m_fileHdr.uiBlockSize);

	// Set up the block table

	m_uiBlockTblSize = 1024;
	if( RC_BAD( rc = f_calloc( 
		sizeof( F_Block *) * m_uiBlockTblSize, &m_ppBlockTbl)))
	{
		m_uiBlockTblSize = 0;
		goto Exit;
	}

	// Set the default container

	m_uiDefaultContainer = FLM_4x_DATA_CONTAINER;;

	// Read the LFile table

	if( RC_BAD( rc = readLFiles()))
	{
		goto Exit;
	}

	// Read the dictionary

	if( RC_BAD( rc = readDictionary()))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		m_pSuperHdl->ReleaseFile( (FLMUINT)0, TRUE);
	}

	if( pucReadBuf)
	{
		f_free( &pucReadBuf);
	}

	if( RC_BAD( rc))
	{
		closeDatabase();
	}

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FLMUINT F_4xReader::lgHdrCheckSum(
	FLMBYTE *	pucLogHdr,
	FLMBOOL		bCompare)
{
	FLMUINT	uiCnt;
	FLMUINT	uiTempSum;
	FLMUINT	uiCurrCheckSum;
	FLMUINT	uiTempSum2;
	FLMUINT	uiBytesToChecksum;

	uiBytesToChecksum = (FB2UW( &pucLogHdr[ 
								FLM_4x_LOG_FLAIM_VERSION]) < FLM_VER_4_3)
									? 88
									: 156;

	if( (uiCurrCheckSum = (FLMUINT)FB2UW( 
		&pucLogHdr[ FLM_4x_LOG_HDR_CHECKSUM])) == 0xFFFF)
	{
		uiCurrCheckSum = 0;
	}

	if( bCompare && !uiCurrCheckSum)
	{
		return( 0);
	}

	for( uiTempSum = 0 - (FLMUINT)FB2UW( &pucLogHdr[ FLM_4x_LOG_HDR_CHECKSUM]),
		uiCnt = 1 + uiBytesToChecksum / sizeof( FLMUINT16); --uiCnt != 0;)
	{
		uiTempSum += (FLMUINT)FB2UW( pucLogHdr);
		pucLogHdr += sizeof( FLMUINT16);
	}

	if( (0 == (uiTempSum2 = (uiTempSum & 0xFFFF))) || (uiTempSum2 == 0xFFFF))
	{
		uiTempSum2 = 1;
	}

	return( (FLMUINT)(((bCompare) && (uiTempSum2 == uiCurrCheckSum))
							? (FLMUINT)0
							: uiTempSum2) );
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::createLckFile(
	char *			pszFilePath)
{
	RCODE				rc = NE_XFLM_OK;
	char				szLockPath[ F_PATH_MAX_SIZE];
	char				szDbBaseName[ F_FILENAME_SIZE];
	char *			pszFileExt;
	F_FileHdl *		pLockFileHdl = NULL;

	// Extract the 8.3 name and put a .lck extension on it to create
	// the full path for the .lck file.

	if( RC_BAD( rc = gv_pFileSystem->pathReduce( 
		pszFilePath, szLockPath, szDbBaseName)))
	{
		goto Exit;
	}

	pszFileExt = &szDbBaseName[ 0];

	while( (*pszFileExt) && (*pszFileExt != '.'))
	{
		pszFileExt++;
	}

	f_strcpy( pszFileExt, ".lck");

	if (RC_BAD( rc = gv_pFileSystem->pathAppend( szLockPath, szDbBaseName)))
	{
		goto Exit;
	}

	if( (pLockFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

#ifndef FLM_UNIX
	pLockFileHdl->setupFileHdl( 0, TRUE);
#else

	// On Unix, we do not want to delete the file because it
	// will succeed even if someone else has the file open.

	pLockFileHdl->setupFileHdl( 0, FALSE);
#endif

	if( RC_BAD( pLockFileHdl->Create( szLockPath,
									XFLM_IO_RDWR | XFLM_IO_EXCL | XFLM_IO_SH_DENYRW)))
	{
#ifndef FLM_UNIX
		if (RC_BAD( gv_pFileSystem->Delete( szLockPath)))
		{
			rc = RC_SET( NE_XFLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		else if (RC_BAD( pLockFileHdl->Create( szLockPath,
									XFLM_IO_RDWR | XFLM_IO_EXCL | XFLM_IO_SH_DENYRW)))
		{
			rc = RC_SET( NE_XFLM_IO_ACCESS_DENIED);
			goto Exit;
		}
#else
		
		if( RC_BAD( pLockFileHdl->Open( szLockPath,
										XFLM_IO_RDWR | XFLM_IO_SH_DENYRW)))
		{
			rc = RC_SET( NE_XFLM_IO_ACCESS_DENIED);
			goto Exit;
		}
#endif
	}

#ifdef FLM_UNIX
	if( RC_BAD( pLockFileHdl->Lock()))
	{
		rc = RC_SET( NE_XFLM_IO_ACCESS_DENIED);
		goto Exit;
	}
#endif

	m_pLckFile = pLockFileHdl;
	pLockFileHdl = NULL;

Exit:

	if (pLockFileHdl)
	{
		pLockFileHdl->Close();
		pLockFileHdl->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
void F_4xReader::closeDatabase( void)
{
	FLMUINT		uiLoop;

	m_tmpPool.poolReset( NULL);

	if( m_pLckFile)
	{
		m_pLckFile->Release();
		m_pLckFile = NULL;
	}

	if( m_pSuperHdl)
	{
		m_pSuperHdl->Release();
	}

	if( m_pLFileTbl)
	{
		f_free( &m_pLFileTbl);
		m_pLFileTbl = NULL;
	}
	m_uiLFileCnt = 0;

	if( m_puiFieldTbl)
	{
		f_free( &m_puiFieldTbl);
		m_puiFieldTbl = NULL;
	}
	m_uiFieldTblSize = 0;

	if( m_ppBlockTbl)
	{
		for( uiLoop = 0; uiLoop < m_uiBlockTblSize; uiLoop++)
		{
			if( m_ppBlockTbl[ uiLoop])
			{
				m_ppBlockTbl[ uiLoop]->Release();
			}
		}

		f_free( &m_ppBlockTbl);
	}
	m_uiBlockTblSize = 0;

	if( m_pNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}

	m_uiMaxFileSize = 0;
	f_memset( &m_fileHdr, 0, sizeof( F_4x_FILE_HDR));
	f_memset( &m_logHdr, 0, sizeof( F_4x_LOG_HDR));
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::readLFiles( void)
{
	RCODE				rc = NE_XFLM_OK;
	F_Block *		pBlock = NULL;
	FLMBYTE *		pucBlk;
	FLMUINT			uiBlkAddress;
	FLMUINT			uiPos;
	FLMUINT			uiEndPos;
	FLMUINT			uiEstCount;
	FLMUINT			uiLFileCnt;
	FLMUINT			uiLFHCnt;
	FLMUINT			uiBlkSize = m_fileHdr.uiBlockSize;
	F_4x_LFILE		TmpLFile;
	F_4x_LFILE *	pLFile;
	F_4x_LFILE *	pLFiles = NULL;

	f_memset( &TmpLFile, 0, sizeof( F_4x_LFILE));

	for( uiEstCount = 0, uiLFileCnt = 4,
		uiBlkAddress = m_fileHdr.uiFirstLFHBlkAddr; 
		uiBlkAddress != FLM_4x_BT_END;)
	{
		if( RC_BAD( rc = readBlock( uiBlkAddress, &pBlock)))
		{
			goto Exit;
		}

		pucBlk = pBlock->m_pucBlk;
		uiPos = FLM_4x_BH_OVHD;

		if( (uiEndPos = (FLMUINT)FB2UW( 
			&pucBlk[ FLM_4x_BH_ELM_END])) <= FLM_4x_BH_OVHD)
		{
			uiEndPos = FLM_4x_BH_OVHD;
			uiLFHCnt = 0;
		}
		else
		{
			if( uiEndPos > uiBlkSize)
			{
				uiEndPos = uiBlkSize;
			}

			uiLFHCnt = (FLMUINT)((uiEndPos - FLM_4x_BH_OVHD) / FLM_4x_LFH_SIZE);
			uiEndPos = (FLMUINT)(FLM_4x_BH_OVHD + uiLFHCnt * FLM_4x_LFH_SIZE);
		}

		// May allocate too many like the inactive ones but OK for now.
		// Allocate an additional 2 for the default data and dict containers.

		if( !uiEstCount)
		{
			uiEstCount = uiLFHCnt + uiLFileCnt;
			if( uiEstCount)
			{
				if( RC_BAD( rc = f_calloc( 
					uiEstCount * sizeof( F_4x_LFILE), &pLFiles)))
				{
					goto Exit;
				}
			}
		}
		else if( uiLFHCnt)
		{
			uiEstCount += uiLFHCnt;

			if( RC_BAD(rc = f_recalloc( uiEstCount * sizeof( F_4x_LFILE),
				&pLFiles)))
			{
				goto Exit;
			}
		}

		// Read through all of the logical file definitions in the block

		for( ; uiPos < uiEndPos; uiPos += FLM_4x_LFH_SIZE)
		{
			FLMUINT	uiLfNum;

			// Have to fix up the offsets later when they are read in

			TmpLFile.uiBlkAddress = uiBlkAddress;
			TmpLFile.uiOffsetInBlk = uiPos;

			if( (TmpLFile.uiLfType = 
				(FLMUINT)pucBlk[ uiPos + FLM_4x_LFH_TYPE_OFFSET]) == FLM_4x_LF_INVALID)
			{
				TmpLFile.uiLfType = FLM_4x_LF_INVALID;
				continue;
			}

			TmpLFile.uiLfNum = 
				(FLMUINT)FB2UW( &pucBlk[ uiPos + FLM_4x_LFH_LF_NUMBER_OFFSET]);

			TmpLFile.uiRootBlk = 
				(FLMUINT)FB2UD( &pucBlk[ uiPos + FLM_4x_LFH_ROOT_BLK_OFFSET]);

			TmpLFile.uiNextDrn = 
				(FLMUINT) FB2UD( &pucBlk[ uiPos + FLM_4x_LFH_NEXT_DRN_OFFSET]);

			uiLfNum = TmpLFile.uiLfNum;

			if( uiLfNum == FLM_4x_DATA_CONTAINER)
			{
				pLFile = pLFiles + FLM_4x_LFILE_DATA_CONTAINER_OFFSET;
			}
			else if( uiLfNum == FLM_4x_DICT_CONTAINER)
			{
				pLFile = pLFiles + FLM_4x_LFILE_DICT_CONTAINER_OFFSET;
			}
			else if( uiLfNum == FLM_4x_DICT_INDEX)
			{
				pLFile = pLFiles + FLM_4x_LFILE_DICT_INDEX_OFFSET;
			}
			else if( uiLfNum == FLM_4x_TRACKER_CONTAINER)
			{
				pLFile = pLFiles + FLM_4x_LFILE_TRACKER_CONTAINER_OFFSET;
			}
			else
			{
				pLFile = pLFiles + uiLFileCnt++;
			}
			f_memcpy( pLFile, &TmpLFile, sizeof( F_4x_LFILE));
		}

		// Get the next block in the chain

		uiBlkAddress = (FLMUINT)FB2UD( &pucBlk[ FLM_4x_BH_NEXT_BLK]);
	}

	m_pLFileTbl = pLFiles;
	m_uiLFileCnt = uiLFileCnt;
	pLFiles = NULL;

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}
	
	if( pLFiles)
	{
		f_free( &pLFiles);
	}

	return( rc );
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getLFile(
	FLMUINT			uiLFile,
	F_4x_LFILE **	ppLFile)
{
	RCODE				rc = NE_XFLM_OK;
	F_4x_LFILE *	pLFile = NULL;
	FLMUINT			uiLoop;

	if( uiLFile == FLM_4x_DATA_CONTAINER)
	{
		pLFile = &m_pLFileTbl[ FLM_4x_LFILE_DATA_CONTAINER_OFFSET];
	}
	else if( uiLFile == FLM_4x_DICT_CONTAINER)
	{
		pLFile = &m_pLFileTbl[ FLM_4x_LFILE_DICT_CONTAINER_OFFSET];
	}
	else if( uiLFile == FLM_4x_TRACKER_CONTAINER)
	{
		pLFile = &m_pLFileTbl[ FLM_4x_LFILE_TRACKER_CONTAINER_OFFSET];
	}
	else
	{
		for( uiLoop = 0; uiLoop < m_uiLFileCnt; uiLoop++)
		{
			if( m_pLFileTbl[ uiLoop].uiLfNum == uiLFile)
			{
				pLFile = &m_pLFileTbl[ uiLoop];
				break;
			}
		}
	}

	if( !pLFile)
	{
		rc = RC_SET( NE_XFLM_BAD_COLLECTION);
		goto Exit;
	}

	*ppLFile = pLFile;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getNextDrn(
	FLMUINT		uiContainer,
	FLMUINT *	puiDrn)
{
	RCODE				rc = NE_XFLM_OK;
	BTSK				stackBuf[ FLM_4x_BH_MAX_LEVELS ];
	BTSK *			pStack = stackBuf;
	FLMBYTE *		pucElm;
	FLMBOOL			bUsedStack = FALSE;
	F_4x_LFILE *	pLFile;

	if( RC_BAD( rc = getLFile( uiContainer, &pLFile)))
	{
		goto Exit;
	}

	bUsedStack = TRUE;
	initStack( &stackBuf[ 0]);

	if( RC_BAD( rc = btSearchEnd( pLFile, 
		FLM_4x_DRN_LAST_MARKER, &pStack)))
	{
		goto Exit;
	}

	if( pLFile->uiRootBlk == FLM_4x_BT_END)
	{
		*puiDrn = pLFile->uiNextDrn;
	}
	else
	{
		if( pLFile->uiLfNum != 
			FB2UW( &pStack->pBlk->m_pucBlk[ FLM_4x_BH_LOG_FILE_NUM]))
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		pucElm = FLM_4x_CURRENT_ELM( pStack);
		pucElm += FLM_4x_BBE_GETR_KL( pucElm) + FLM_4x_BBE_KEY;
		*puiDrn = FB2UD( pucElm);
	}

Exit:

	if( bUsedStack)
	{
		releaseStack( stackBuf);
	}

	return( rc);
}

/***************************************************************************
Desc:		Search the right-most end of the b-tree.
****************************************************************************/
RCODE F_4xReader::btSearchEnd(
	F_4x_LFILE *	pLFile,
	FLMUINT			uiDrn,
	BTSK **			ppStack)
{
	RCODE			rc = NE_XFLM_OK;
	BTSK *		pStack = *ppStack;
	FLMBYTE		ucKey[ FLM_4x_DIN_KEY_SIZ];
	FLMUINT		uiBlkAddr;

	if( RC_BAD( rc = getRootBlock( pLFile, pStack)))
	{
		goto Exit;
	}

	longToByte( uiDrn, ucKey);
	for(;;)
	{
		if( pStack->uiLevel)
		{
			pStack->uiCurElm = pStack->uiBlkEnd;
			btPrevElm( pStack, pLFile);
		}
		else
		{
			if( pStack->uiBlkType != FLM_4x_BHT_NON_LEAF_DATA)
			{
				if( RC_BAD( rc = btScan( pStack, ucKey)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = btScanNonLeafData( pStack, uiDrn)))
				{
					goto Exit;
				}
			}
		}

		if( !pStack->uiLevel)
		{
			break;
		}

		uiBlkAddr = childBlkAddr( pStack);
		pStack++;

		if( RC_BAD( rc = getBlock( pLFile, uiBlkAddr, pStack)))
		{
			goto Exit;
		}
	}

	*ppStack = pStack;

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::readBlock(
	FLMUINT			uiBlkAddr,
	F_Block **		ppBlock)
{
	RCODE			rc = NE_XFLM_OK;
	F_Block *	pBlock = NULL;
	F_Block *	pReuseBlock = NULL;
	F_Block **	ppTblSlot = NULL;

	if( *ppBlock)
	{
		(*ppBlock)->Release();
		*ppBlock = NULL;
	}

	if( m_ppBlockTbl)
	{
		ppTblSlot = getHashBucket( uiBlkAddr);
		pBlock = *ppTblSlot;

		if( pBlock)
		{
			if( FLM_4x_GET_BH_ADDR( pBlock->m_pucBlk) != uiBlkAddr)
			{
				if( pBlock->getRefCount() == 1)
				{
					pReuseBlock = *ppTblSlot;
				}
				else
				{
					(*ppTblSlot)->Release();
				}

				pBlock = NULL;
				*ppTblSlot = NULL;
			}
			else
			{
				pBlock->AddRef();
			}
		}
	}

	if( !pBlock)
	{
		if( pReuseBlock)
		{
			pBlock = pReuseBlock;
			pReuseBlock = NULL;
		}
		else
		{
			if( (pBlock = f_new F_Block) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}
		}

		if( RC_BAD( rc = pBlock->allocBlockBuf( m_fileHdr.uiBlockSize)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pSuperHdl->ReadBlock( uiBlkAddr,
			m_fileHdr.uiBlockSize, pBlock->m_pucBlk, NULL)))
		{
			if( rc == NE_XFLM_IO_END_OF_FILE)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
			}
			goto Exit;
		}

		// Verify the block checksum

		if( RC_BAD( rc = blkCheckSum( 
			pBlock->m_pucBlk, uiBlkAddr, m_fileHdr.uiBlockSize)))
		{
			goto Exit;
		}

		// See if we even got the block we thought we wanted

		if( FLM_4x_GET_BH_ADDR( pBlock->m_pucBlk) != uiBlkAddr)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		flmAssert( *ppTblSlot == NULL);
		*ppTblSlot = pBlock;
		pBlock->AddRef();
	}

	flmAssert( *ppTblSlot == pBlock);
	*ppBlock = pBlock;
	pBlock = NULL;

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	if( pReuseBlock)
	{
		pReuseBlock->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::blkCheckSum(
	FLMBYTE *	pucBlkPtr,
	FLMUINT		uiBlkAddress,
	FLMUINT		uiBlkSize)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiCnt;
	FLMUINT			uiAdds;
	FLMUINT			uiXORs;
	FLMUINT			uiCurrCheckSum;
	FLMUINT			uiNewCheckSum;
	FLMUINT			uiEncryptSize;
	FLMBYTE *		pucSaveBlkPtr = pucBlkPtr;

	// Check the block length against the max. block size

	uiEncryptSize = (FLMUINT)getEncryptSize( pucBlkPtr);
	if( uiEncryptSize > uiBlkSize || uiEncryptSize < FLM_4x_BH_OVHD)
	{
		rc = RC_SET( NE_XFLM_BLOCK_CRC);
		goto Exit;
	}	

	uiCurrCheckSum = (FLMUINT)(((FLMUINT)pucBlkPtr[ 
				FLM_4x_BH_CHECKSUM_HIGH] << 8) + 
				(FLMUINT)pucBlkPtr[ FLM_4x_BH_CHECKSUM_LOW]);

	uiAdds = 0 - (pucBlkPtr[ FLM_4x_BH_CHECKSUM_LOW] + 
					pucBlkPtr[ FLM_4x_BH_CHECKSUM_HIGH]);

	uiXORs = pucBlkPtr[ FLM_4x_BH_CHECKSUM_LOW] ^ 
					pucBlkPtr[ FLM_4x_BH_CHECKSUM_HIGH];

	if( uiBlkAddress != FLM_4x_BT_END)
	{
		uiAdds += (FLMBYTE)uiBlkAddress;
		uiXORs ^= (FLMBYTE)uiBlkAddress;
	}

	for( uiCnt = uiEncryptSize; uiCnt--;)
	{
		uiAdds += *pucBlkPtr;
		uiXORs ^= *(pucBlkPtr++);
	}

	uiNewCheckSum = (((uiAdds << 8) + uiXORs) & 0xFFFF);
	
	if( uiBlkAddress == FLM_4x_BT_END )
	{
		FLMBYTE		byXor;
		FLMBYTE		byAdd;
		FLMBYTE		byDelta;
		
		// If there is a one byte value that will satisfy both
		// sides of the checksum, the checksum is OK and that value
		// is the first byte value.
		
		byXor = (FLMBYTE) uiNewCheckSum;
		byAdd = (FLMBYTE) (uiNewCheckSum >> 8);
		byDelta = byXor ^ pucSaveBlkPtr[ FLM_4x_BH_CHECKSUM_LOW];
		
		// This is the big check, if byDelta is also what is
		// off with the add portion of the checksum, we have
		// a good value.
		
		if( ((FLMBYTE) (byAdd + byDelta)) == 
			pucSaveBlkPtr[ FLM_4x_BH_CHECKSUM_HIGH])
		{
			// Set the low checksum value with the computed value.
			
			pucSaveBlkPtr[ FLM_4x_BH_CHECKSUM_LOW] = byDelta;
			goto Exit;
		}
	}
	else
	{
		// This has the side effect of setting the low block address byte
		// in the block thus getting rid of the low checksum byte.
		//
		// NOTE: We are allowing the case where the calculated checksum is
		// zero and the stored checksum is one because we used to change
		// a calculated zero to a one in old databases and store the one.
		// This is probably a somewhat rare case (1 out of 65536 checksums
		// will be zero), so forgiving it will be OK most of the time.
		// So that those don't cause us to report block checksum errors,
		// we just allow it - checksumming isn't a perfect check anyway.
		
		if( uiNewCheckSum == uiCurrCheckSum ||
			 ((!uiNewCheckSum) && (uiCurrCheckSum == 1)))
		{
			pucSaveBlkPtr[ FLM_4x_BH_CHECKSUM_LOW] = (FLMBYTE)uiBlkAddress;
			goto Exit;
		}
	}
	
	// Otherwise, we have a checksum error.
	
	rc = RC_SET( NE_XFLM_BLOCK_CRC);

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::retrieveRec(
	FLMUINT			uiContainer,
	FLMUINT			uiDrn,
	FLMUINT			uiFlags,
	F_Record **		ppRecord)
{
	RCODE				rc = NE_XFLM_OK;
	BTSK				stackBuf[ FLM_4x_BH_MAX_LEVELS];
	BTSK *			pStack = NULL;
	F_4x_LFILE *	pLFile;

	initStack( &stackBuf[ 0]);
	pStack = stackBuf;

	if( uiDrn >= (FLM_4x_DRN_LAST_MARKER - 1))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = getLFile( uiContainer, &pLFile)))
	{
		goto Exit;
	}

	if( uiFlags & XFLM_INCL)
	{
		// Search the for the record

		if( RC_BAD( rc = btSearch( pLFile, uiDrn, &pStack)))
		{
			goto Exit;
		}

		if( byteToLong( pStack->ucKeyBuf) == FLM_4x_DRN_LAST_MARKER)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		flmAssert( byteToLong( pStack->ucKeyBuf) >= uiDrn);
	}
	else if( uiFlags & XFLM_EXCL)
	{
		// Search the for the record

		if( RC_BAD( rc = btSearch( pLFile, uiDrn + 1, &pStack)))
		{
			goto Exit;
		}

		if( byteToLong( pStack->ucKeyBuf) == FLM_4x_DRN_LAST_MARKER)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		flmAssert( byteToLong( pStack->ucKeyBuf) > uiDrn);
	}
	else
	{
		// Search the for the record

		if( RC_BAD( rc = btSearch( pLFile, uiDrn, &pStack)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = RC_SET( NE_XFLM_NOT_FOUND);
			}

			goto Exit;
		}

		if( !pStack->uiKeyLen ||
			byteToLong( pStack->ucKeyBuf) != uiDrn)
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
			goto Exit;
		}
	}

	// Read the record

	if( RC_BAD( rc = readRecElements( pStack, pLFile, ppRecord)))
	{
		goto Exit;
	}

Exit:

	releaseStack( stackBuf);
	return( rc);
}

/***************************************************************************
Desc:		
*****************************************************************************/
RCODE F_4xReader::retrieveNextRec(
	F_Record **		ppRecord)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiDrn = 0;
	FLMUINT			uiContainer = m_uiDefaultContainer;

	if( *ppRecord)
	{
		uiDrn = (*ppRecord)->getID();
		uiContainer = (*ppRecord)->getContainerID();
	}

	if( RC_BAD( rc = retrieveRec( uiContainer, uiDrn, XFLM_EXCL, ppRecord)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/**************************************************************************
Desc:		
**************************************************************************/
void F_4xReader::releaseStack(
	BTSK *		pStack)
{
	FLMUINT		uiNumLevels = FLM_4x_BH_MAX_LEVELS;

	while( uiNumLevels)
	{
		if( pStack->pBlk)
		{
			pStack->pBlk->Release();
			pStack->pBlk = NULL;
		}
		uiNumLevels--;
		pStack++;
	}
}

/**************************************************************************
Desc:		
**************************************************************************/
RCODE F_4xReader::readRecElements(
	BTSK *			pStack,
	F_4x_LFILE *	pLFile,
	F_Record **		ppRecord)
{
	RCODE				rc = NE_XFLM_OK;
	F_Record *		pRecord = NULL;
	FLMBYTE *		pucCurElm;
	void  *			pvMark = m_tmpPool.poolMark();
	FLMUINT			uiElmRecLen;
	FLMUINT			uiFieldLen;
	FLMUINT			uiFieldCount;
	FLMUINT			uiTrueDataSpace;
	FLMUINT			uiFieldPos;
	TFIELD *			pField;
	FLDGROUP *		pFldGroup = NULL;	
	FLDGROUP *		pFirstFldGroup = NULL;
	DATAPIECE *		pDataPiece;
	LOCKED_BLOCK * pLockedBlock = NULL;
	FSTATE			state;

	// Initialize variables

	state.uiLevel = 0;
	uiFieldCount = 0;
	uiTrueDataSpace = 0;
	uiFieldPos = NUM_FIELDS_IN_ARRAY;

	// Check to make sure we are positioned at the first element.

	pucCurElm = FLM_4x_CURRENT_ELM( pStack);

	if( !FLM_4x_BBE_IS_FIRST( pucCurElm))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Loop on each element in the record

	for( ;;)
	{
		// Setup all variables to process the current element

		uiElmRecLen = FLM_4x_BBE_GET_RL( pucCurElm);

		if( !uiElmRecLen)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			break;
		}

		pucCurElm += FLM_4x_BBE_REC_OFS( pucCurElm);
		state.uiPosInElm = 0;

		// Loop on each field within this element.

		while( state.uiPosInElm < uiElmRecLen)
		{
			state.pElement = pucCurElm;
			if( RC_BAD( rc = getFldOverhead( &state)))
			{
				goto Exit;
			}

			uiFieldLen = state.uiFieldLen;

			// Old record info data - skip past for now

			if( !state.uiTagNum)
			{
				state.uiPosInElm += uiFieldLen;
				continue;
			}

			if( !pRecord)
			{
				// Create a new data record or use the existing data record.

				if( *ppRecord)
				{
					// Reuse the existing F_Record object.

					pRecord = *ppRecord;
					*ppRecord = NULL;
					pRecord->clear();
				}
				else
				{
					if( (pRecord = f_new F_Record) == NULL)
					{
						rc = RC_SET( NE_XFLM_MEM);
						goto Exit;
					}
				}

				pRecord->setContainerID( 
						FB2UW( &pStack->pBlk->m_pucBlk[ FLM_4x_BH_LOG_FILE_NUM]));
				pRecord->setID( byteToLong( pStack->ucKeyBuf));
			}

			// Check if out of fields in the tempoary field group

			if( uiFieldPos >= NUM_FIELDS_IN_ARRAY)
			{
				FLDGROUP *		pTempFldGroup;

				uiFieldPos = 0;

				// Allocate the first field group from the pool.

				if( RC_BAD( rc = m_tmpPool.poolAlloc( 
					sizeof( FLDGROUP), (void **)&pTempFldGroup)))
				{
					goto Exit;
				}

				pTempFldGroup->pNext = NULL;
				if( pFldGroup)
				{
					pFldGroup->pNext = pTempFldGroup;
				}
				else
				{
					pFirstFldGroup = pTempFldGroup;
				}
				pFldGroup = pTempFldGroup;
			}
			
			flmAssert( state.uiFieldType != FLM_4x_UNKNOWN_TYPE);
			uiFieldCount++;
			pField = &pFldGroup->pFields[ uiFieldPos++];
			pField->uiLevel = state.uiLevel;
			pField->uiFieldID = state.uiTagNum;
			pField->uiFieldType = state.uiFieldType;
			pField->uiFieldLen = state.uiFieldLen;
			pDataPiece = &pField->DataPiece;

			if( uiFieldLen)
			{
				FLMUINT		uiDataPos = 0;

				if( state.uiFieldLen > 4)
				{
					// Binary data needs to account for alignment issues.

					if( state.uiFieldType == FLM_4x_BINARY_TYPE)
					{
						if( state.uiFieldLen >= 255)
						{
							// Align so that the data is aligned - not the length

							uiTrueDataSpace += 2;
							uiTrueDataSpace = ((uiTrueDataSpace + FLM_ALLOC_ALIGN) &
											(~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
							uiTrueDataSpace -= 2;
						}
						else
						{
							uiTrueDataSpace = ((uiTrueDataSpace + FLM_ALLOC_ALIGN) & 
													(~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
						}
					}

					uiTrueDataSpace += state.uiFieldLen;

					// For read only records, greater than 255 bytes are
					// stored length preceded.

					if( state.uiFieldLen >= 255)
					{
						uiTrueDataSpace += 2;
					}
				}

				// Value may start in the next element.

				while( uiDataPos < uiFieldLen)
				{
					// Need to read next element for the value portion? 

					if( state.uiPosInElm >= uiElmRecLen)
					{
						if( FLM_4x_BBE_IS_LAST( FLM_4x_CURRENT_ELM( pStack)))
						{
							rc = RC_SET( NE_XFLM_DATA_ERROR);
							goto Exit;
						}

						// If we are going to the next block, lock down this block
						// beacause data pointers are pointing to it.

						if( RC_BAD( blkNextElm( pStack)))
						{
							LOCKED_BLOCK * pLastLockedBlock = pLockedBlock;
								
							if( RC_BAD( rc = m_tmpPool.poolAlloc( 
								sizeof( LOCKED_BLOCK), (void **)&pLockedBlock)))
							{
								goto Exit;
							}

							pLockedBlock->pBlock = pStack->pBlk;
							pLockedBlock->pBlock->AddRef();
							pLockedBlock->pNext = pLastLockedBlock;

							if( RC_BAD( rc = btNextElm( pStack, pLFile)))
							{
								goto Exit;
							}
						}

						pucCurElm = FLM_4x_CURRENT_ELM( pStack);
						uiElmRecLen = FLM_4x_BBE_GET_RL( pucCurElm);
						pucCurElm += FLM_4x_BBE_REC_OFS( pucCurElm);
						state.uiPosInElm = 0;
					}

					// Compare number of bytes left if value <= # bytes left in element

					if( (uiFieldLen - uiDataPos) <= 
						(uiElmRecLen - state.uiPosInElm))
					{
						FLMUINT	uiDelta = uiFieldLen - uiDataPos;
						
						pDataPiece->pData = &pucCurElm[ state.uiPosInElm];
						pDataPiece->uiLength = uiDelta;
						state.uiPosInElm += uiDelta;
						pDataPiece->pNext = NULL;
						break;
					}
					else
					{
						// Take what is there and get next element to grab some more.

						FLMUINT		uiBytesToMove = uiElmRecLen - state.uiPosInElm;
						DATAPIECE *	pNextDataPiece;

						pDataPiece->pData = &pucCurElm[ state.uiPosInElm];
						pDataPiece->uiLength = uiBytesToMove;
						state.uiPosInElm += uiBytesToMove;
						uiDataPos += uiBytesToMove;

						if( RC_BAD( rc = m_tmpPool.poolAlloc( 
							sizeof( DATAPIECE), (void **)&pNextDataPiece)))
						{
							goto Exit;
						}
						pDataPiece->pNext = pNextDataPiece;
						pDataPiece = pNextDataPiece;
					}
				}
			}
		}

		// Done?

		if( FLM_4x_BBE_IS_LAST( FLM_4x_CURRENT_ELM( pStack)))
		{
			break;
		}

		// Position to next element

		if( RC_BAD( blkNextElm( pStack)))
		{
			LOCKED_BLOCK *	pLastLockedBlock = pLockedBlock;
				
			if( RC_BAD( rc = m_tmpPool.poolAlloc( 
				sizeof( LOCKED_BLOCK), (void **)&pLockedBlock)))
			{
				goto Exit;
			}

			pLockedBlock->pBlock = pStack->pBlk;
			pLockedBlock->pBlock->AddRef();
			pLockedBlock->pNext = pLastLockedBlock;

			if( RC_BAD( rc = btNextElm( pStack, pLFile)))
			{
				goto Exit;
			}
		}

		// Corruption Check.

		pucCurElm = FLM_4x_CURRENT_ELM( pStack);
		if( FLM_4x_BBE_IS_FIRST( pucCurElm))
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
	}

	if( pRecord)
	{
		void *		pvField;

		if( RC_BAD( rc = pRecord->preallocSpace( uiTrueDataSpace)))
		{
			goto Exit;
		}
		pFldGroup = pFirstFldGroup;
		
		for( uiFieldPos = 0; uiFieldCount--; uiFieldPos++)
		{

			if( uiFieldPos >= NUM_FIELDS_IN_ARRAY)
			{
				uiFieldPos = 0;
				if( (pFldGroup = pFldGroup->pNext) == NULL)
				{
					break;
				}
			}
			pField = &pFldGroup->pFields[ uiFieldPos];

			if( RC_BAD( rc = pRecord->insertLast( pField->uiLevel, pField->uiFieldID,
					pField->uiFieldType, &pvField)))
			{
				goto Exit;
			}

			if( pField->uiFieldLen)
			{
				FLMBYTE *	pDataPtr;			// Points to where the data will go.

				pDataPiece = &pField->DataPiece;
				pDataPtr = pRecord->getImportDataPtr( pvField, 
					pField->uiFieldType, pField->uiFieldLen);

				if( !pDataPtr)
				{
					rc = RC_SET( NE_XFLM_MEM);
					goto Exit;
				}

				do
				{
					f_memcpy( pDataPtr, pDataPiece->pData, pDataPiece->uiLength);
					pDataPtr += pDataPiece->uiLength;
					pDataPiece = pDataPiece->pNext;
				} 
				while( pDataPiece);
			}
		}
	}

	if( *ppRecord)
	{
		flmAssert( 0);
		(*ppRecord)->Release();
	}
	
	*ppRecord = pRecord;
	pRecord = NULL;
	
Exit:

	// Release all locked down blocks except the current block.

	while( pLockedBlock)
	{
		pLockedBlock->pBlock->Release();
		pLockedBlock = pLockedBlock->pNext;
	}

	m_tmpPool.poolReset( pvMark);

	if( pRecord)
	{
		pRecord->Release();
	}

	return( rc);
}

/**************************************************************************
Desc:		
**************************************************************************/
RCODE F_4xReader::getFldOverhead(
	FSTATE *			pState)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pFieldOvhd = &pState->pElement[ pState->uiPosInElm];
	FLMBYTE *	pElement = pState->pElement;
	FLMBOOL		bDoesntHaveFieldDef = TRUE;
	FLMUINT		uiFieldLen;	
	FLMUINT		uiFieldType = 0;
	FLMUINT		uiTagNum;
	FLMBYTE		ucTemp;

	if( FLM_4x_FOP_IS_STANDARD( pFieldOvhd))
	{
		if( FLM_4x_FSTA_LEVEL( pFieldOvhd))
		{
			pState->uiLevel++;
		}

		uiFieldLen = FLM_4x_FSTA_FLD_LEN( pFieldOvhd);
		uiTagNum = FLM_4x_FSTA_FLD_NUM( pFieldOvhd);
		pFieldOvhd += FLM_4x_FSTA_OVHD;
	}
	else if( FLM_4x_FOP_IS_OPEN( pFieldOvhd))
	{
		if( FLM_4x_FOPE_LEVEL( pFieldOvhd))
		{
			pState->uiLevel++;
		}

		ucTemp = (FLMBYTE)(FLM_4x_FOP_GET_FLD_FLAGS( pFieldOvhd++));
		uiTagNum = (FLMUINT)*pFieldOvhd++;

		if( FLM_4x_FOP_2BYTE_FLDNUM( ucTemp))
		{
			uiTagNum += ((FLMUINT) *pFieldOvhd++) << 8;
		}

		uiFieldLen = (FLMUINT) *pFieldOvhd++;
		if( FLM_4x_FOP_2BYTE_FLDLEN( ucTemp))
		{
			uiFieldLen += ((FLMUINT) *pFieldOvhd++) << 8;
		}
	}
	else if( FLM_4x_FOP_IS_NO_VALUE( pFieldOvhd))
	{
		if( FLM_4x_FNOV_LEVEL( pFieldOvhd))
		{
			pState->uiLevel++;
		}

		ucTemp = (FLMBYTE)(FLM_4x_FOP_GET_FLD_FLAGS( pFieldOvhd++));
		uiTagNum = (FLMUINT)*pFieldOvhd++;

		if( FLM_4x_FOP_2BYTE_FLDNUM( ucTemp))
		{
			uiTagNum += ((FLMUINT) *pFieldOvhd++) << 8;
		}
		uiFieldLen = uiFieldType = 0;
	}
	else if( FLM_4x_FOP_IS_SET_LEVEL( pFieldOvhd))
	{
		pState->uiLevel -= FLM_4x_FSLEV_GET( pFieldOvhd++);
		pState->uiPosInElm = (FLMUINT)( pFieldOvhd - pElement);
		rc = getFldOverhead( pState);
		goto Exit;
	}
	else if( FLM_4x_FOP_IS_TAGGED( pFieldOvhd))
	{
		bDoesntHaveFieldDef = FALSE;

		if( FLM_4x_FTAG_LEVEL( pFieldOvhd))
		{
			pState->uiLevel++;
		}

		ucTemp = (FLMBYTE)(FLM_4x_FOP_GET_FLD_FLAGS( pFieldOvhd));
		uiFieldType = (FLMUINT)(FLM_4x_FTAG_FLD_TYPE( pFieldOvhd));
		pFieldOvhd += FLM_4x_FTAG_OVHD;
		uiTagNum = (FLMUINT) *pFieldOvhd++;

		if( FLM_4x_FOP_2BYTE_FLDNUM( ucTemp))
		{
			uiTagNum += ((FLMUINT) *pFieldOvhd++) << 8;
		}
	
		uiTagNum ^= 0x8000;
		uiFieldLen = (FLMUINT)*pFieldOvhd++;
		if( FLM_4x_FOP_2BYTE_FLDLEN( ucTemp))
		{
			uiFieldLen += ((FLMUINT) *pFieldOvhd++) << 8;
		}
	}
	else if( FLM_4x_FOP_IS_RECORD_INFO( pFieldOvhd))
	{
		bDoesntHaveFieldDef = FALSE;
		ucTemp = *pFieldOvhd++;
		uiFieldLen = *pFieldOvhd++;

		if( FLM_4x_FOP_2BYTE_FLDLEN( ucTemp))
		{
			uiFieldLen += ((FLMUINT) *pFieldOvhd++) << 8;
		}
		uiTagNum = 0;
	}
	else
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	if( bDoesntHaveFieldDef)
	{
		if( RC_BAD( rc = getFieldType( uiTagNum, &uiFieldType)))
		{
			goto Exit;
		}
	}

	pState->uiFieldType = uiFieldType;
	pState->uiFieldLen = uiFieldLen;
	pState->uiPosInElm = (FLMUINT)(pFieldOvhd - pElement);
	pState->uiTagNum = uiTagNum;

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
FLMUINT F_4xReader::childBlkAddr(
	BTSK *			pStack)
{
	FLMBYTE *		pucChildBlkPtr;
	FLMUINT			uiElmOvhd = pStack->uiElmOvhd;
			
	if( uiElmOvhd == FLM_4x_BNE_DATA_OVHD)
	{
		pucChildBlkPtr = FLM_4x_BLK_ELM_ADDR( 
			pStack, pStack->uiCurElm + FLM_4x_BNE_DATA_CHILD_BLOCK);
		return( FB2UD( pucChildBlkPtr));
	}
	else
	{
		// Corruption
			
		flmAssert( 0);
		return( 0);
	}
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::btSearch(
	F_4x_LFILE *	pLFile,
	FLMUINT			uiDrn,
	BTSK **			ppStack)
{
	RCODE				rc = NE_XFLM_OK;
	BTSK *			pStack = *ppStack;
	FLMBYTE			ucKey[ FLM_4x_DIN_KEY_SIZ];
	FLMUINT			uiBlkAddr;

	// Get the root block

	if( RC_BAD( rc = getRootBlock( pLFile, pStack)))
	{
		goto Exit;
	}

	longToByte( uiDrn, ucKey);

	// Read each block going down the b-tree.
	// Save state information in the stack.
	
	for(;;)
	{
		if( pStack->uiBlkType != FLM_4x_BHT_NON_LEAF_DATA)
		{
			if( RC_BAD( rc = btScan( pStack, ucKey)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = btScanNonLeafData( pStack, uiDrn)))
			{
				goto Exit;
			}
		}
		
		if( !pStack->uiLevel)
		{
			break;
		}

		uiBlkAddr = childBlkAddr( pStack);
		pStack++;

		if( RC_BAD( rc = getBlock( pLFile, uiBlkAddr, pStack)))
		{
			goto Exit;
		}
	}

	*ppStack = pStack;

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Scan a b-tree block for a matching key at any b-tree block level.
****************************************************************************/
RCODE F_4xReader::btScan(
	BTSK *			pStack,
	FLMBYTE *		pucSearchKey)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucCurElm;
	FLMBYTE *		pBlk;
	FLMBYTE *		pucKeyBuf;
	FLMBYTE *		pElmKey;
	FLMUINT			uiRecLen = 0;
	FLMUINT			uiPrevKeyCnt;
	FLMUINT			uiElmKeyLen;
	FLMUINT			uiBlkType;
	FLMUINT			uiElmOvhd;
	FLMUINT			uiBytesMatched;

	uiBlkType = pStack->uiBlkType;
	flmAssert( uiBlkType != FLM_4x_BHT_NON_LEAF_DATA);

	pucKeyBuf = pStack->ucKeyBuf;
	pBlk = pStack->pBlk->m_pucBlk;
	uiElmOvhd = pStack->uiElmOvhd;
	pStack->uiCurElm = FLM_4x_BH_OVHD;
	pStack->uiKeyLen = 0;
	pStack->uiPKC = 0;
	pStack->uiPrevElmPKC = 0;
	uiBytesMatched = 0;

	for( ;;)
	{
		pucCurElm = &pBlk[ pStack->uiCurElm];
		uiElmKeyLen = FLM_4x_BBE_GETR_KL( pucCurElm);

		// Read in RAW mode - doesn't do all bit checking 
		
		if( (uiPrevKeyCnt = (FLM_4x_BBE_GETR_PKC( pucCurElm))) > 
			FLM_4x_BBE_PKC_MAX)
		{
			uiElmKeyLen += (uiPrevKeyCnt & FLM_4x_BBE_KL_HBITS) << 
									FLM_4x_BBE_KL_SHIFT_BITS;
			uiPrevKeyCnt &= FLM_4x_BBE_PKC_MAX;
		}

		// Should not have a non-zero PKC if we are on the first element
		// of a block

		if( uiPrevKeyCnt && pStack->uiCurElm == FLM_4x_BH_OVHD)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		// Get the record portion length when on the leaf blocks.
		
		if( uiBlkType == FLM_4x_BHT_LEAF)
		{
			uiRecLen = FLM_4x_BBE_GET_RL( pucCurElm);
		}

		pStack->uiPrevElmPKC = pStack->uiPKC;
		
		// The zero length key is the terminating 
		// element in a right-most block.

		if( (pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen) == 0)
		{
			pStack->uiPrevElmPKC = f_min( uiBytesMatched, FLM_4x_BBE_PKC_MAX);
			pStack->uiPKC = 0;
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		// Handle special case of left-end compression maxing out.

		if( uiPrevKeyCnt == FLM_4x_BBE_PKC_MAX && 
			FLM_4x_BBE_PKC_MAX < uiBytesMatched)
		{
			uiBytesMatched = FLM_4x_BBE_PKC_MAX;
		}

		// Check out this element to see if the key matches.

		if( uiPrevKeyCnt == uiBytesMatched)
		{
			pElmKey = &pucCurElm[ uiElmOvhd];

			for(;;)
			{
				// All bytes of the search key are matched?

				if( uiBytesMatched == FLM_4x_DIN_KEY_SIZ)
				{
					pStack->uiPKC = f_min( uiBytesMatched, FLM_4x_BBE_PKC_MAX);

					if( pStack->uiKeyLen != FLM_4x_DIN_KEY_SIZ)
					{
						rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
						goto Exit;
					}

					// Current key is equal to the search key.

					f_memcpy( pucKeyBuf, pucSearchKey, FLM_4x_DIN_KEY_SIZ);
					goto Exit;
				}

				if( uiBytesMatched == pStack->uiKeyLen)
				{
					pStack->uiPKC = f_min( uiBytesMatched, FLM_4x_BBE_PKC_MAX);
					goto Next_Element;
				}

				// Compare the next byte in the search key and element

				if( pucSearchKey[ uiBytesMatched] != *pElmKey)
				{
					break;
				}

				uiBytesMatched++;
				pElmKey++;
			}

			pStack->uiPKC = f_min( uiBytesMatched, FLM_4x_BBE_PKC_MAX);

			// Check if we are done comparing

			if( pucSearchKey[ uiBytesMatched] < *pElmKey)
			{
				if( uiBytesMatched)
				{
					flmAssert( uiBytesMatched <= FLM_4x_DIN_KEY_SIZ);
					f_memcpy( pucKeyBuf, pucSearchKey, uiBytesMatched);
				}

				flmAssert( pStack->uiKeyLen <= FLM_4x_DIN_KEY_SIZ);
				f_memcpy( &pucKeyBuf[ uiBytesMatched], pElmKey, 
								pStack->uiKeyLen - uiBytesMatched);
				goto Exit;
			}
		}
		else if( uiPrevKeyCnt < uiBytesMatched)
		{
			// Current key > search key.  Set pucKeyBuf and break out.

			pStack->uiPKC = uiPrevKeyCnt;
			if( uiPrevKeyCnt)
			{
				flmAssert( uiPrevKeyCnt <= FLM_4x_DIN_KEY_SIZ);
				f_memcpy( pucKeyBuf, pucSearchKey, uiPrevKeyCnt);
			}

			flmAssert( uiPrevKeyCnt + uiElmKeyLen <= FLM_4x_DIN_KEY_SIZ);
			f_memcpy( &pucKeyBuf[ uiPrevKeyCnt], 
				&pucCurElm[ uiElmOvhd], uiElmKeyLen);

			if( byteToLong( pucKeyBuf) == FLM_4x_BT_END)
			{
				rc = RC_SET( NE_XFLM_EOF_HIT);
			}

			goto Exit;
		}

Next_Element:

		// Position to the next element

		pStack->uiCurElm += uiElmKeyLen + ((uiBlkType == FLM_4x_BHT_LEAF )
						? (FLM_4x_BBE_KEY + uiRecLen)
						: (FLM_4x_BNE_IS_DOMAIN( pucCurElm) 
							? (FLM_4x_BNE_DOMAIN_LEN + uiElmOvhd)
							: uiElmOvhd));
		
		// Most common check first.

		if( pStack->uiCurElm < pStack->uiBlkEnd)
		{
			continue;
		}

		if( pStack->uiCurElm == pStack->uiBlkEnd)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		// Marched off the end of the block

		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Binary search into a non-leaf data record block.
****************************************************************************/
RCODE F_4xReader::btScanNonLeafData(
	BTSK *			pStack,
	FLMUINT			uiDrn)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucBlk = pStack->pBlk->m_pucBlk;
	FLMUINT		uiLow = 0;
	FLMUINT		uiMid;
	FLMUINT		uiHigh = ((pStack->uiBlkEnd - FLM_4x_BH_OVHD) >> 3) - 1;
	FLMUINT		uiTblSize = uiHigh;
	FLMUINT		uiCurDrn;

	for(;;)
	{
		uiMid = (uiLow + uiHigh) >> 1;
		
		uiCurDrn = byteToLong( &pucBlk[ FLM_4x_BH_OVHD + (uiMid << 3)]);
		if( !uiCurDrn)
		{
			// Special case - at the end of a rightmost block.
			break;
		}

		if( uiDrn == uiCurDrn)
		{
			// Remember a data record can span multiple blocks (same DRN).

			while( uiMid)
			{
				uiCurDrn = byteToLong( 
					&pucBlk[ FLM_4x_BH_OVHD + ((uiMid - 1) << 3)]);
				if( uiDrn != uiCurDrn)
				{
					break;
				}
				uiMid--;
			}
			break;
		}

		// Down to one item if too high then position to next item.

		if( uiLow >= uiHigh)
		{
			if( (uiDrn > uiCurDrn) && uiMid < uiTblSize)
			{
				uiMid++;
			}

			break;
		}

		// If too high then try lower section

		if( uiDrn < uiCurDrn)
		{
			// First item too high?

			if( !uiMid)
			{
				break;
			}

			uiHigh = uiMid - 1;
		}
		else
		{
			if( uiMid == uiTblSize)
			{
				uiMid++;
				rc = RC_SET( NE_XFLM_EOF_HIT);
				goto Exit;
			}

			uiLow = uiMid + 1;
		}
	}

	// Set curElm and the key buffer.

	pStack->uiCurElm = FLM_4x_BH_OVHD + (uiMid << 3);
	longToByte( uiCurDrn, pStack->ucKeyBuf);

Exit:

	return( rc);
}

/****************************************************************************
Desc:  	Goto the next element within the block
****************************************************************************/
RCODE F_4xReader::blkNextElm(
	BTSK *			pStack)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucElm;
	FLMUINT	 		uiElmSize;

	pucElm = &pStack->pBlk->m_pucBlk[ pStack->uiCurElm];

	if( pStack->uiBlkType == FLM_4x_BHT_LEAF)
	{	
		uiElmSize = FLM_4x_BBE_LEN( pucElm);
		if( pStack->uiCurElm + FLM_4x_BBE_LEM_LEN < pStack->uiBlkEnd)
		{
			if( ((pStack->uiCurElm += uiElmSize) + 
				FLM_4x_BBE_LEM_LEN < pStack->uiBlkEnd) == 0)
			{
				rc = RC_SET( NE_XFLM_EOF_HIT);
				goto Exit;
			}

		}
		else
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}
	}
	else
	{
		if( pStack->uiBlkType != FLM_4x_BHT_NON_LEAF_DATA)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		uiElmSize = FLM_4x_BNE_DATA_OVHD;

		if( pStack->uiCurElm < pStack->uiBlkEnd)
		{
			// Check if this is not the last element within the block

			if( (pStack->uiCurElm += uiElmSize) >= pStack->uiBlkEnd)
			{
				rc = RC_SET( NE_XFLM_EOF_HIT);
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Go to the next element in the logical b-tree
****************************************************************************/
RCODE F_4xReader::btNextElm(
	BTSK *			pStack,
	F_4x_LFILE *	pLFile)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucCurElm;
	FLMUINT		uiLFile;

	uiLFile = FB2UW( &pStack->pBlk->m_pucBlk[ FLM_4x_BH_LOG_FILE_NUM]);
	 
	 if( pStack->uiCurElm < FLM_4x_BH_OVHD)
	{
		pStack->uiCurElm = FLM_4x_BH_OVHD;
	}
	else
	{
		if( RC_BAD( rc = blkNextElm( pStack)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				FLMBYTE *		pucBlk = FLM_4x_BLK_ELM_ADDR( pStack, FLM_4x_BH_NEXT_BLK);
				FLMUINT			uiBlkNum = FB2UD( pucBlk);

				if( uiBlkNum != FLM_4x_BT_END)
				{
					// Current element was last element in the block - goto next block */

					if( RC_BAD( rc = getBlock( pLFile, uiBlkNum, pStack)))
					{
						goto Exit;
					}
			
					pucBlk = pStack->pBlk->m_pucBlk;
					pStack->uiBlkEnd = (FLMUINT)FB2UW( &pucBlk[ FLM_4x_BH_ELM_END ]);
					pStack->uiCurElm = FLM_4x_BH_OVHD;
					btAdjustStack( pStack, pLFile, TRUE);
				}
			}
		}
	}

	pucCurElm = FLM_4x_CURRENT_ELM( pStack);

	// Copy the key

	f_memcpy( pStack->ucKeyBuf, pucCurElm, FLM_4x_DIN_KEY_SIZ);

Exit:	

	return(rc );
}

/***************************************************************************
Desc:		Go to the previous element in the logical b-tree
****************************************************************************/
RCODE F_4xReader::btPrevElm(
	BTSK *			pStack,
	F_4x_LFILE *	pLFile)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiBlkAddr;
	FLMUINT		uiTargetElm;
	FLMUINT		uiPrevElm = 0;
	FLMUINT		uiPrevKeyCnt = 0;
	FLMUINT		uiElmKeyLen = 0;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMBYTE *	pucCurElm;
	FLMBYTE *	pucBlk;

	// Check if we are at or before the first element in the block

	if( pStack->uiCurElm <= FLM_4x_BH_OVHD)
	{
		pucBlk = pStack->pBlk->m_pucBlk;

		// We're at or before the first element, so read in the previous
		// block and go to the last element

		if( (uiBlkAddr = (FLMUINT)FB2UD( &pucBlk[ 
			FLM_4x_BH_PREV_BLK])) == FLM_4x_BT_END)
		{
			// We are at the end

			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}
		else
		{
			if( RC_BAD( rc = getBlock( pLFile, uiBlkAddr, pStack)))
			{
				// Set uiBlkEnd and uiCurElm.
				// Adjust the parent block to the previous element

				pucBlk = pStack->pBlk->m_pucBlk;
				pStack->uiCurElm = pStack->uiBlkEnd;
				btAdjustStack( pStack, pLFile, FALSE);
				goto Exit;
			}
		}
	}

	// Move down 1 before the current element

	if( pStack->uiBlkType == FLM_4x_BHT_NON_LEAF_DATA)
	{
		pStack->uiCurElm -= FLM_4x_BNE_DATA_OVHD;
		pucBlk = pStack->pBlk->m_pucBlk;
		pucCurElm = &pucBlk[ pStack->uiCurElm];
		f_memcpy( pStack->ucKeyBuf, pucCurElm, FLM_4x_DIN_KEY_SIZ);
		goto Exit;
	}

	// Set up to point to first element in the block

	uiTargetElm = pStack->uiCurElm;
	pStack->uiCurElm = FLM_4x_BH_OVHD;
	pucBlk = pStack->pBlk->m_pucBlk;
	
	while( pStack->uiCurElm < uiTargetElm)
	{
		pucCurElm = &pucBlk[ pStack->uiCurElm];
		uiPrevKeyCnt = (FLMUINT)(FLM_4x_BBE_GET_PKC( pucCurElm));
		uiElmKeyLen  = (FLMUINT)(FLM_4x_BBE_GET_KL( pucCurElm));

		if( uiElmKeyLen + uiPrevKeyCnt > FLM_4x_DIN_KEY_SIZ)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		if( uiElmKeyLen)
		{
			flmAssert( uiPrevKeyCnt + uiElmKeyLen <= FLM_4x_DIN_KEY_SIZ);
			f_memcpy( &pStack->ucKeyBuf[ uiPrevKeyCnt], 
						 &pucCurElm[ uiElmOvhd], uiElmKeyLen);
		}

		uiPrevElm = pStack->uiCurElm;
		if( RC_BAD( rc = blkNextElm( pStack)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}
			rc = NE_XFLM_OK;
			break;
		}
	}

	pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen;
	pStack->uiCurElm = uiPrevElm;
	flmAssert( pStack->uiKeyLen == FLM_4x_DIN_KEY_SIZ);

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::btAdjustStack(
	BTSK *			pStack,
	F_4x_LFILE *	pLFile,
	FLMBOOL			bMovedNext)
{
	RCODE			rc = NE_XFLM_OK;

	pStack--;
	if( RC_BAD( rc = getBlock( pLFile, pStack->uiBlkAddr, pStack)))
	{
		goto Exit;
	}

	if( bMovedNext)
	{
		if( RC_BAD( rc = btNextElm( pStack, pLFile)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = btPrevElm( pStack, pLFile)))
		{
			goto Exit;
		}
	}

Exit:

	pStack++;
	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getNameTable(
	F_4xNameTable **		ppNameTable)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bAllocated = FALSE;

	if( !m_pNameTable)
	{
		if( (m_pNameTable = f_new F_4xNameTable) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		bAllocated = TRUE;

		if( RC_BAD( rc = m_pNameTable->setupNameTable( this)))
		{
			goto Exit;
		}
	}

	flmAssert( *ppNameTable == NULL);
	m_pNameTable->AddRef();
	*ppNameTable = m_pNameTable;

Exit:

	if( RC_BAD( rc) && bAllocated)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
F_Record::F_Record() 
{
	m_uiContainerID = 0;
	m_uiRecordID = 0;
	m_pool.poolInit( sizeof( m_fieldList));
	m_pFirstFld = m_pLastFld = NULL;
	resetFieldList();
	m_pDataBuf = NULL;
	m_uiDataBufOffset = 0;
	m_uiDataBufLength = 0;
}

/***************************************************************************
Desc:		
****************************************************************************/
F_Record::~F_Record()
{
	m_pool.poolFree();
	if( m_pDataBuf)
	{
		f_free( &m_pDataBuf);
	}
}

/***************************************************************************
Desc:		
****************************************************************************/
void F_Record::resetFieldList( void)
{
	FLMUINT	uiLoop;
	FIELD *	pCurField;
	FIELD *	pPrevField = NULL;

 	for( uiLoop = 0; uiLoop < FLM_4x_FIELD_LIST_SIZE; uiLoop++)
	{
		pCurField = &m_fieldList[ uiLoop];
		pCurField->ui16FieldID = 0xFFFF;

		pCurField->pPrev = pPrevField;
		pCurField->pNext = &m_fieldList[ uiLoop + 1];
		pPrevField = pCurField;
	}

	pCurField->pNext = NULL;

	m_pAvailFld = &m_fieldList[ 0];
}

/***************************************************************************
Desc:		
****************************************************************************/
void F_Record::clear()
{
	m_uiDataBufOffset = 0;
	resetFieldList();

	m_pool.poolReset( NULL);
	m_pFirstFld = NULL;
	m_pLastFld = NULL;

	m_uiContainerID = 0;
	m_uiRecordID = 0;
}

/***************************************************************************
Desc:		
****************************************************************************/
FIELD * F_Record::lastSubTreeField(
	FIELD *		pField)
{
	FIELD *	pTempField = (FIELD *)lastChild( pField);
	FIELD *	pLastChild = NULL;
	FLMUINT	uiStartLevel = pField->ui8Level;

	// Step down through the tree 

	for( ; pTempField && pTempField->ui8Level > uiStartLevel;
			pTempField = nextField( pTempField))
	{
		pLastChild = pTempField;
	}

	return( pLastChild);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::insertLast(
	FLMUINT		uiLevel,
	FLMUINT		uiFieldID,
	FLMUINT		uiDataType,
	void **		ppvField)
{
	RCODE		rc = NE_XFLM_OK;
	FIELD *	pField = NULL;
	
	// Insert new field following current last field

	if( RC_BAD( rc = createField( m_pLastFld, &pField)))
	{
		goto Exit;
	}

	// Set up the new field and set as the current field

	pField->ui16FieldID = (FLMUINT16) uiFieldID;
	pField->ui8Level = (FLMUINT8) uiLevel;
	pField->ui8Type = (FLMUINT8) uiDataType;

	if( ppvField)
	{
		*ppvField = pField;
	}

Exit:

#ifdef FLM_DEBUG
	if ( pField)
	{
		flmAssert( pField->pNext != pField);
		flmAssert( pField->pPrev != pField);
		flmAssert( pField->ui16FieldID != 0xFFFF);
	}
#endif

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
FIELD * F_Record::nextSiblingField(
	FIELD *		pField)
{
	FLMUINT8		ui8Level = pField->ui8Level;

	while( (pField = nextField( pField)) != NULL && 
		pField->ui8Level > ui8Level)
	{
		;
	}

	return( (pField && pField->ui8Level == ui8Level)
					? pField
					: NULL);
}

/***************************************************************************
Desc:		
****************************************************************************/
void * F_Record::prevSibling(
	void *		pvField)
{
	if( !pvField)
	{
		return( NULL);
	}

	FIELD *	pField = (FIELD *)pvField;
	FLMUINT8 ui8Level = pField->ui8Level;

	while( (pField = prevField( pField)) != NULL &&
		pField->ui8Level > ui8Level)
	{
		;
	}

	return( (pField && pField->ui8Level == ui8Level)
					? pField
					: NULL);
}

/***************************************************************************
Desc:		
****************************************************************************/
void * F_Record::lastChild( 
	void *		pvField)
{
	FIELD *	pField = (FIELD *)pvField;
	FIELD *	pLastField = NULL;

	if( !pField)
	{
		return( NULL);
	}

	for( pField = firstChildField( pField); 
		pField; 
		pField = nextSiblingField( pField))
	{
		pLastField = pField;
	}

	return( pLastField);
}

/***************************************************************************
Desc:		
****************************************************************************/
void * F_Record::parent(
	void *		pvField)
{
	FIELD *	pField = (FIELD *) pvField;
	FLMUINT8 ui8Level;

	if( !pField)
	{
		return( NULL);
	}

	ui8Level = pField->ui8Level;

	while( (pField = prevField( pField)) != NULL && 
		pField->ui8Level >= ui8Level)
	{
		;
	}

	return( pField);
}

/***************************************************************************
Desc:		
****************************************************************************/
FLMBYTE * F_Record::getImportDataPtr( 
	void *		pvField,
	FLMUINT		uiDataType,
	FLMUINT		uiLength) 
{
	FLMBYTE *	pucData = NULL;

	getNewDataPtr( (FIELD *)pvField, uiDataType, uiLength, &pucData);
	((FIELD *)pvField)->ui8Type = (FLMUINT8) uiDataType;

	return( pucData);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::preallocSpace(
	FLMUINT	uiDataSize)
{
	RCODE		rc = NE_XFLM_OK;

	m_uiDataBufOffset = 0;
	if( m_uiDataBufLength >= uiDataSize)
	{
		goto Exit;
	}

	if( !m_pDataBuf)
	{
		m_uiDataBufLength = 0;
		if( RC_BAD( rc = f_alloc( uiDataSize, &m_pDataBuf)))
		{
			goto Exit;
		}
		m_uiDataBufLength = uiDataSize;
	}
	else
	{
		if( RC_BAD( rc = f_realloc( uiDataSize, &m_pDataBuf)))
		{
			goto Exit;
		}
		m_uiDataBufLength = uiDataSize;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
void * F_Record::find(
	void *	pvField,
	FLMUINT	uiFieldID,
	FLMUINT	uiOccur,
	FLMBOOL	bSearchForest)
{
	if( uiOccur == 0)
	{
		uiOccur = 1;
	}

	if( pvField)
  	{
		FLMUINT	uiStartLevel = ((FIELD *)pvField)->ui8Level;
		do
		{
			if( (uiFieldID == 
				((FIELD *)pvField)->ui16FieldID) && (--uiOccur < 1))
			{
				return( pvField);
			}
		} 
		while( (pvField = (FIELD *)(pvField 
						? ((FIELD *)pvField)->pNext 
						: NULL)) != NULL && 
								((((FIELD *)pvField)->ui8Level > uiStartLevel) || 
								(bSearchForest && ((FIELD *)pvField)->ui8Level ==
								uiStartLevel)));
	}

	return( NULL);
}

/***************************************************************************
Desc:		
****************************************************************************/
void * F_Record::find(
	void *		pvField,
	FLMUINT *	puiPathArray,
	FLMUINT		uiOccur,
	FLMBOOL		bSearchForest)
{
	void *		pvSaveField;
	FLMUINT *	puiPath;
	FLMUINT		uiLevel;

	// Handle empty record

	if( !pvField)
	{
		return( NULL);
	}

	if( !uiOccur)
	{
		uiOccur = 1;
	}

	uiLevel = ((FIELD *)pvField)->ui8Level;
	for(;;)
	{
		puiPath = puiPathArray + ( ((FIELD *)pvField)->ui8Level - uiLevel);
		pvSaveField = pvField;

		if( *puiPath == ((FIELD *)pvField)->ui16FieldID)
		{
			if( *(puiPath + 1) == 0 && (--uiOccur < 1))
			{
				return( pvField);
			}

			// Go down level for rest of path

			if( ( pvField = firstChild( pvField)) != NULL)
			{
				continue;									
			}
			pvField = pvSaveField;
		}

		// Find next sibling/uncle/end

		do
		{
			pvField = (FIELD *)(pvField ? ((FIELD *)pvField)->pNext : NULL);
		}
		while( pvField != NULL 
			&& ((FIELD *)pvField)->ui8Level > ((FIELD *)pvSaveField)->ui8Level);

		// Are we at the end?

		if( !pvField ||
			((FIELD *)pvField)->ui8Level < uiLevel ||
			(bSearchForest && ((FIELD *)pvField)->ui8Level == uiLevel)) 
		{
			break;
		}
	}

	return( NULL);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE	F_Record::createField(
	FIELD *		pCurField,
	FIELD **		ppNewField)
{
	RCODE			rc = NE_XFLM_OK;
	FIELD *		pNewField;

	if( m_pAvailFld)
	{
		pNewField = m_pAvailFld;
		m_pAvailFld = m_pAvailFld->pNext;
		if( m_pAvailFld)
			m_pAvailFld->pPrev = NULL;
		pNewField->pPrev = NULL;
		pNewField->pNext = NULL;
	}
	else
	{
		if( RC_BAD( rc = m_pool.poolAlloc( sizeof( FIELD), (void **)&pNewField)))
		{
			goto Exit;
		}
	}

	if( !pCurField && m_pLastFld)
	{
		pCurField = m_pLastFld;
	}

	if( pCurField)
	{
		pNewField->pNext = pCurField->pNext;
		pNewField->pPrev = pCurField;
		
		if( pCurField->pNext)
		{
			pCurField->pNext->pPrev = pNewField;
		}
		pCurField->pNext = pNewField;
	}

	if( !m_pFirstFld)
	{
		m_pFirstFld = pNewField;
	}

	if( !m_pLastFld || pCurField == m_pLastFld)
	{
		m_pLastFld = pNewField;
	}

	pNewField->uiDataLength = 0;
	pNewField->uiDataOffset = 0;
	*ppNewField = pNewField;

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE	F_Record::getNewDataPtr(
	FIELD *		pField, 
	FLMUINT		uiDataType,
	FLMUINT		uiNewLength, 
	FLMBYTE **	ppDataPtr)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pDataPtr;
	FLMUINT		uiTemp;

	if( pField->uiDataLength)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	if( uiNewLength <= sizeof( FLMUINT))
	{
		pDataPtr = (FLMBYTE *) &(pField->uiDataOffset);
	}
	else
	{
		// If this is a binary field it must start on an aligned byte.

		if( uiDataType == FLM_4x_BINARY_TYPE && 
			(m_uiDataBufOffset & FLM_ALLOC_ALIGN) != 0)
		{
			uiTemp = (FLM_ALLOC_ALIGN + 1) - (m_uiDataBufOffset & FLM_ALLOC_ALIGN);
			m_uiDataBufOffset += uiTemp;
		}

		if( uiNewLength + m_uiDataBufOffset > m_uiDataBufLength)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}

		pDataPtr = m_pDataBuf + m_uiDataBufOffset;
		pField->uiDataOffset = m_uiDataBufOffset;
		m_uiDataBufOffset += uiNewLength;
	}

	pField->uiDataLength = uiNewLength;
	*ppDataPtr = pDataPtr;

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getINT(
	void *		pvField,
	FLMINT *		piNumber)
{
	return( pvField
		? storage2INT( getDataType( pvField), getDataLength( pvField),
					getDataPtr( (FIELD *)pvField), piNumber)
		: RC_SET( NE_XFLM_NOT_FOUND));
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getUINT(
	void *		pvField,
	FLMUINT *	puiNumber)
{
	return( pvField
		? storage2UINT( getDataType( pvField), getDataLength( pvField),
					getDataPtr( (FIELD *)pvField), puiNumber)
		: RC_SET( NE_XFLM_NOT_FOUND));
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getUINT32(
	void *		pvField,
	FLMUINT32 *	pui32Number)
{
	return( pvField
		? storage2UINT32( getDataType( pvField), getDataLength( pvField),
					getDataPtr( (FIELD *)pvField), pui32Number)
		: RC_SET( NE_XFLM_NOT_FOUND));
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getUnicode( 
	void *			pvField,
	FLMUNICODE *	pUnicode, 
	FLMUINT *		puiBufLen)
{
	return( pvField
		? getUnicode( getDataType( pvField), getDataLength( pvField),
				getDataPtr( (FIELD *) pvField), puiBufLen, pUnicode)
		: RC_SET( NE_XFLM_NOT_FOUND));
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getNative(
	void *		pvField,
	char *		pszString, 
	FLMUINT *	puiBufLen)
{
	return( pvField
		? storage2Native( getDataType( pvField), getDataLength( pvField),
					getDataPtr( (FIELD *) pvField), puiBufLen, pszString)
		: RC_SET( NE_XFLM_NOT_FOUND));
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getBinary( 
	void *		pvField,
	void *		pvBuf,
	FLMUINT *	puiBufLen)
{
	if( pvField)
	{
		*puiBufLen = f_min( *puiBufLen, getDataLength( pvField));
		f_memcpy( pvBuf, getDataPtr( (FIELD *)pvField), *puiBufLen);
		return( NE_XFLM_OK);
	}

	return( RC_SET( NE_XFLM_NOT_FOUND));
}

/***************************************************************************
Desc:		
****************************************************************************/
FLMBYTE * F_Record::getDataPtr(
	FIELD *		pField) 
{
	if( !pField->uiDataLength)
	{
		return( NULL);
	}
	else if( pField->uiDataLength <= sizeof( FLMUINT))
	{
		return (FLMBYTE *) &(pField->uiDataOffset);
	}

	return( m_pDataBuf + pField->uiDataOffset);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::storage2INT(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuf,
	FLMINT *		piNum)
{
	RCODE			rc = NE_XFLM_OK;
	BCD_TYPE		bcd;

	if( RC_BAD( rc = bcd2Num( uiType, uiBufLength, pBuf, &bcd)))
	{
		goto Exit;
	}

	if( bcd.bNegFlag)
	{
		*piNum = -((FLMINT)bcd.uiNum);
		if( !((bcd.uiNibCnt < 11) ||
			(bcd.uiNibCnt == 11 && 
				(!bcd.pucPtr || (f_memcmp( bcd.pucPtr, 
				gv_ucMinBcdINT32, 6) <= 0)))))
		{
			rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
			goto Exit;
		}
	}
	else
	{
		*piNum = (FLMINT)bcd.uiNum;

		if( !((bcd.uiNibCnt < 10) ||
			(bcd.uiNibCnt == 10 && 
				(!bcd.pucPtr || (f_memcmp( bcd.pucPtr, 
				gv_ucMaxBcdINT32, 5) <= 0)))))
		{
			rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::storage2UINT(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuf,
	FLMUINT *	puiNum)
{
	RCODE			rc = NE_XFLM_OK;
	BCD_TYPE		bcd;

	if( RC_BAD( rc = bcd2Num( uiType, uiBufLength, pBuf, &bcd)))
	{
		goto Exit;
	}

	*puiNum = bcd.uiNum;
	
	if( bcd.bNegFlag)
	{
		rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
		goto Exit;
	}
	else if( bcd.uiNibCnt == 10) 
	{
		if( !(!bcd.pucPtr || 
			(f_memcmp( bcd.pucPtr, gv_ucMaxBcdUINT32, 5) <= 0)))
		{
			rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
			goto Exit;
		}
	}
	else if( bcd.uiNibCnt > 10)
	{
		rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::storage2UINT32(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuf,
	FLMUINT32 *	pui32Num)
{
	RCODE			rc = NE_XFLM_OK;
	BCD_TYPE		bcd;

	if( RC_BAD( rc = bcd2Num( uiType, uiBufLength, pBuf, &bcd)))
	{
		goto Exit;
	}

	*pui32Num = (FLMUINT32)bcd.uiNum;
	
	if( bcd.bNegFlag)
	{
		rc = RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW);
		goto Exit;
	}
	else if( bcd.uiNibCnt == 10) 
	{
		if( !(!bcd.pucPtr || 
			(f_memcmp( bcd.pucPtr, gv_ucMaxBcdUINT32, 5) <= 0)))
		{
			rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
			goto Exit;
		}
	}
	else if( bcd.uiNibCnt > 10)
	{
		rc = RC_SET( NE_XFLM_CONV_NUM_OVERFLOW);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::bcd2Num(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuf,
	BCD_TYPE *	bcd)
{
	RCODE		rc = NE_XFLM_OK;

	if( !pBuf)
	{
		rc = RC_SET( NE_XFLM_CONV_NULL_SRC);
		goto Exit;
	}

	switch( uiType)
	{
		case FLM_4x_NUMBER_TYPE:
		{
			FLMUINT 		uiTotalNum = 0;
			FLMUINT		uiByte;
			FLMUINT		uiNibCnt;

			bcd->pucPtr = pBuf;

			// Get each nibble and use to create the number

#define FLM_MAX_NIB_CNT		11

			for( bcd->bNegFlag = 
				(FLMBOOL)(uiNibCnt = ((*pBuf & 0xF0) == 0xB0) ? 1 : 0);
				uiNibCnt <= FLM_MAX_NIB_CNT;
				uiNibCnt++ )
			{
				uiByte = (uiNibCnt & 0x01)
						? (FLMUINT)(0x0F & *pBuf++)
						: (FLMUINT)(*pBuf >> 4);

				if( uiByte == 0x0F)
				{
					break;
				}

				// Multiply by 10 and add n
				// NOTE: 10y = 8y + 2y = (y << 3) + (y << 1)
				// faster than using the long multiply (10 * y)

				uiTotalNum = (uiTotalNum << 3) + (uiTotalNum << 1) + uiByte;
			}

			bcd->uiNibCnt = uiNibCnt;
			bcd->uiNum = uiTotalNum;
			break;
		}

		case FLM_4x_TEXT_TYPE:
		{
			FLMUINT		uiNumber = 0;

			// If it is a TEXT Value, convert to a numeric value
			// WARNING: The text is not null terminated

			while( uiBufLength--)
			{
				if( *pBuf < ASCII_ZERO || *pBuf > ASCII_NINE)
				{
					break;
				}

				uiNumber = (uiNumber * 10) + (*pBuf - ASCII_ZERO);
				pBuf++;
			}

			bcd->uiNum = uiNumber;
			bcd->uiNibCnt = 0;
			bcd->bNegFlag = FALSE;
			break;
		}

		case FLM_4x_CONTEXT_TYPE :
		{
			if( uiBufLength == sizeof( FLMUINT32))
			{
				bcd->uiNum = (FLMUINT)( FB2UD( pBuf));
				bcd->bNegFlag = 0;

				// Now set the uiNibCnt, the uiNibCnt will not be totally
				// accurate, but it's close enough to get the value out...

				if( bcd->uiNum < FLM_MAX_UINT8)
				{
					bcd->uiNibCnt = 3;
				}
				else if( bcd->uiNum < FLM_MAX_UINT16)
				{
					bcd->uiNibCnt = 5;
				}
				else
				{
					bcd->uiNibCnt = 9;
				}
			}

			break;
		}

		default :
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:		
****************************************************************************/
RCODE F_Record::getUnicode(
	FLMUINT			uiType,
	FLMUINT			uiBufLength,
	FLMBYTE *		pBuffer,
	FLMUINT *		puiStrBufLen,
	FLMUNICODE *	puzStrBuf)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucChar;
	FLMUINT			uiBytesProcessed = 0;
	FLMUINT			uiBytesOutput = 0;
	FLMBOOL			bOutputData = FALSE;
	FLMUINT			uiMaxOutLen;
	FLMBYTE			ucObjType;
	FLMUINT			uiObjLength = 0;
	FLMBYTE			tempBuf[ 80];
	FLMBYTE			chrSet, chrVal;
	FLMUNICODE		newChrVal;

	// If the value is a number, convert to text first

	if( uiType != FLM_4x_TEXT_TYPE)
	{
		if( !pBuffer)
		{
			uiBufLength = 0;
		}
		else
		{
			if( uiType == FLM_4x_NUMBER_TYPE)
			{
				uiBufLength = sizeof( tempBuf);
				if( RC_BAD( rc = numToText( pBuffer, tempBuf, &uiBufLength)))
				{
					goto Exit;
				}
			}
			else if( uiType == FLM_4x_TEXT_TYPE)
			{
				uiBufLength = sizeof( tempBuf);
				if( RC_BAD( rc = contextToText( pBuffer, tempBuf, &uiBufLength)))
				{
					goto Exit;
				}
			}
			else	
			{
				rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
				goto Exit;
			}

			pBuffer = &tempBuf[ 0];
		}
	}

	uiMaxOutLen = *puiStrBufLen;
	if( puzStrBuf != NULL && uiMaxOutLen > 1)
	{
		bOutputData = TRUE;
		uiMaxOutLen -= 2;
	}

	// Parse through the string outputting data to the buffer as we go

	while( uiBytesProcessed < uiBufLength)
	{
		// Determine what we are pointing at

		ucChar = *pBuffer;
		ucObjType = (FLMBYTE)textObjType( ucChar);

		switch( ucObjType)
		{
			case FLM_4x_ASCII_CHAR_CODE:
			{
				uiObjLength = 1;
				if( bOutputData)
				{
					if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto GetUNICODE_Output;
					}
					*puzStrBuf++ = ucChar;
				}
				uiBytesOutput += 2;
				break;
			}

			case FLM_4x_CHAR_SET_CODE:
			{
				uiObjLength = 2;
				if( bOutputData)
				{
					if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto GetUNICODE_Output;
					}

					// Convert WP to UNICODE
					
					chrSet = ucChar & 0x3F;
					chrVal = *(pBuffer + 1);

					if( RC_BAD( rc = flmWPToUnicode( 
						(((FLMUINT16)chrSet) << 8) | chrVal, &newChrVal)))
					{
						RC_UNEXPECTED_ASSERT( rc);
						goto Exit;
					}

					if( bOutputData)
					{
						if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
						{
							rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
							goto GetUNICODE_Output;
						}
						*puzStrBuf++ = newChrVal; 
					}
				}

				uiBytesOutput += 2;
				break;
			}
				
			case FLM_4x_WHITE_SPACE_CODE:
			{
				uiObjLength = 1;

				if( bOutputData)
				{
					if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto GetUNICODE_Output;
					}

					if( ucChar == (FLM_4x_WHITE_SPACE_CODE | 0x0C))
					{
						*puzStrBuf = 9;
					}
					else if( ucChar == (FLM_4x_WHITE_SPACE_CODE | 0x0D))
					{
						*puzStrBuf = 10;
					}
					else if( ucChar == (FLM_4x_WHITE_SPACE_CODE | 0x07))
					{
						*puzStrBuf = 13;
					}
					else
					{
						*puzStrBuf = 0x20;
					}
					puzStrBuf++;
				}
				uiBytesOutput += 2;
				break;
			}

			case FLM_4x_EXT_CHAR_CODE:
			{
				uiObjLength = 3;
				if( bOutputData)
				{
					if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto GetUNICODE_Output;
					}
					
					// Convert back from WP to UNICODE

					chrSet = *(pBuffer + 1);
					chrVal = *(pBuffer + 2);

					if( RC_BAD( rc = flmWPToUnicode( 
						(((FLMUINT16)chrSet) << 8) | chrVal, &newChrVal)))
					{
						RC_UNEXPECTED_ASSERT( rc);
						goto Exit;
					}

					if( bOutputData)
					{
						if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
						{
							rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
							goto GetUNICODE_Output;
						}
						*puzStrBuf++ = newChrVal; 
					}
				}
				uiBytesOutput += 2;		

				break;
			}

			case FLM_4x_OEM_CODE:
			{
				uiObjLength = 2;
				break;
			}

			case FLM_4x_UNICODE_CODE:
			{
				uiObjLength = 3;
				if( bOutputData)
				{
					if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto GetUNICODE_Output;
					}
					*puzStrBuf++ = (*(pBuffer + 1) << 8) + *(pBuffer + 2);
				}
				uiBytesOutput += 2;
				break;
			}
	
			case FLM_4x_UNK_EQ_1_CODE:
			{
				uiObjLength = 2;
				if( bOutputData)
				{
					if( (uiMaxOutLen < 2) || (uiBytesOutput > uiMaxOutLen - 2))
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto GetUNICODE_Output;
					}
					*puzStrBuf++ = *(pBuffer+1);
				}
				uiBytesOutput += 2;
				break;
			}

			default:
			{
				flmAssert( 0);
				uiBytesProcessed = uiBufLength;
				break;
			}
		}

		pBuffer += uiObjLength;
		uiBytesProcessed += uiObjLength;
	}

GetUNICODE_Output:

	if( bOutputData)
	{
		*puzStrBuf = 0;
	}

	*puiStrBufLen = uiBytesOutput;

Exit:

	return( rc);
}

/***************************************************************************
Desc: 	Convert a storage text string into a native string
***************************************************************************/
RCODE F_Record::storage2Native(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuffer, 
	FLMUINT *	puiOutBufLenRV,
	char *		pOutBuffer)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	ptr = pBuffer;
	FLMBYTE *	pucOut;
	FLMBYTE		ucChar;
	FLMUINT		uiBytesProcessed;
	FLMUINT		uiBytesOutput;
	FLMUINT		uiValLength = uiBufLength;
	FLMBOOL		bOutputData = FALSE;
	FLMUINT		uiMaxOutLen = 0;
	FLMBYTE		ucObjType;
	FLMUINT		uiObjLength = 0;
	FLMBYTE		TempBuf[ 80];

	// If needed, try and convert the data to text

	if( uiType != FLM_4x_TEXT_TYPE)
	{
		if( !ptr)
		{
			uiValLength = 0;
		}
		else if( uiType == FLM_4x_NUMBER_TYPE)
		{
			uiValLength = sizeof( TempBuf);
			if( RC_BAD( rc = numToText( ptr, TempBuf, &uiValLength)))
			{
				goto Exit;
			}
			ptr = &TempBuf[ 0];
		}
		else if( uiType == FLM_4x_CONTEXT_TYPE)
		{
			uiValLength = sizeof( TempBuf);
			if( RC_BAD( rc = contextToText( ptr, TempBuf, &uiValLength)))
			{
				goto Exit;
			}
			ptr = &TempBuf[ 0];
		}
		else
		{
			rc = RC_SET( NE_XFLM_CONV_ILLEGAL);
			goto Exit;
		}
	}

	if( pOutBuffer != NULL && *puiOutBufLenRV)
	{
		bOutputData = TRUE;
		uiMaxOutLen = *puiOutBufLenRV - 1;
	}

	uiBytesProcessed = 0;
	uiBytesOutput = 0;
	pucOut = (FLMBYTE *)pOutBuffer;

	// Parse through the string outputting data to the buffer
	// as we go

	while( uiBytesProcessed < uiValLength)
	{
		// Determine what we are pointing at

		ucChar = *ptr;
		ucObjType = (FLMBYTE)textObjType( ucChar);
		switch( ucObjType)
		{
			case FLM_4x_ASCII_CHAR_CODE:
			{
				uiObjLength = 1;
				if( bOutputData)
				{
					if( uiBytesOutput < uiMaxOutLen)
						*pucOut++ = f_tonative( ucChar);
					else
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				uiBytesOutput++;
				break;
			}

			case FLM_4x_CHAR_SET_CODE:
			{
				uiObjLength = 2;
				if( bOutputData)
				{
					if( uiBytesOutput < uiMaxOutLen)
					{
						if( (ucChar & (~ucObjType)) == 0)
						{
							*pucOut++ = f_tonative( *(ptr + 1));
						}
						else
						{
							*pucOut++ = 0xFF;
						}
					}
					else
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				uiBytesOutput++;
				break;
			}

			case FLM_4x_WHITE_SPACE_CODE:
			{
				uiObjLength = 1;

				if( bOutputData)
				{
					if( uiBytesOutput < uiMaxOutLen)
					{
						ucChar &= (~FLM_4x_WHITE_SPACE_MASK);
						if(	(ucChar == 0x03) ||
								(ucChar == 0x04) ||
								(ucChar == 0x05))
						{
							ucChar = ASCII_DASH;
						}
						else if( ucChar == 0x0C)
						{
							ucChar = ASCII_TAB;
						}
						else if( ucChar == 0x0D)
						{
							ucChar = 0x0A;
						}
						else if( ucChar == 0x07)
						{
							ucChar = 0x0D;
						}
						else
						{
							ucChar = 0x20;
						}

						*pucOut++ = f_tonative( ucChar);
					}
					else
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				uiBytesOutput++;
				break;
			}

			case FLM_4x_UNK_EQ_1_CODE:
			{
				uiObjLength = 2;

				// Skip it if it is not a NATIVE code

				if( (ucChar & (~ucObjType)) == 0x02)
				{
					if( bOutputData)
					{
						if( uiBytesOutput < uiMaxOutLen)
						{
							*pucOut++ = f_tonative( *(ptr + 1));
						}
						else
						{
							rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
							goto Native_Output;
						}
					}
					uiBytesOutput++;
				}
				break;
			}

			case FLM_4x_EXT_CHAR_CODE:
			{
				uiObjLength = 3;
				if( bOutputData)
				{
					if( uiBytesOutput < uiMaxOutLen)
					{
						*pucOut += 0xFF;
					}
					else
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				uiBytesOutput++;
				break;
			}

			case FLM_4x_OEM_CODE:
			{
				uiObjLength = 2;
				if( bOutputData)
				{
					if( uiBytesOutput < uiMaxOutLen)
					{
						*pucOut++ = f_tonative( *(ptr + 1));
					}
					else
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				uiBytesOutput++;
				break;
			}

			case FLM_4x_UNICODE_CODE:
			{
				uiObjLength = 3;
				if( bOutputData)
				{
					if( uiBytesOutput < uiMaxOutLen )
					{
						*pucOut++ = 0xFF;
					}
					else
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				uiBytesOutput++;
				break;
			}

			default:
			{
				flmAssert( 0);
				break;
			}
		}

		ptr += uiObjLength;
		uiBytesProcessed += uiObjLength;
	}

	// Add a terminating NULL character, but DO NOT increment the
	// uiBytesOutput counter

Native_Output:

	if( bOutputData)
	{
		*pucOut = 0;
	}

	*puiOutBufLenRV = uiBytesOutput;

Exit:

	return( rc);
}

/***************************************************************************
Desc: 	Convert a storage number into a storage text string
***************************************************************************/
RCODE F_Record::numToText(
	FLMBYTE *	pucNum,
	FLMBYTE *	pucOutBuffer,
	FLMUINT *	puiBufLen)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucOutput;
	FLMBYTE		ucChar;
	FLMBYTE		ucOutChar;
	FLMUINT		uiBytesOutput;
	FLMBOOL		bOutputData;
	FLMUINT		uiMaxOutLen;
	FLMBOOL		bFirstNibble;

	uiMaxOutLen = *puiBufLen;
	bOutputData = ((pucOutBuffer != NULL) && uiMaxOutLen) ? TRUE : FALSE;
	uiBytesOutput = 0;
	pucOutput = pucOutBuffer;

	// Parse through the string outputting data to the buffer
	// as we go

	if( !pucNum)
	{
		goto Exit;
	}

	bFirstNibble = TRUE;
	for( ;;)
	{
		if( bFirstNibble)
		{
			ucChar = (FLMBYTE)(*pucNum >> 4);
		}
		else
		{
			ucChar  = (FLMBYTE)(*pucNum++ & 0x0F);
		}

		bFirstNibble = !bFirstNibble;

		if( ucChar <= 9)
		{
			ucOutChar = (FLMBYTE)( ASCII_ZERO + ucChar);
		}
		else if( ucChar == 0x0F)
		{
			break;
		}
		else
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}

		if( bOutputData)
		{
			if( uiBytesOutput < uiMaxOutLen)
			{
				*pucOutput++ = ucOutChar;
			}
			else
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}
		}

		uiBytesOutput++;
	}

Exit:

	*puiBufLen = uiBytesOutput;
	return( rc);
}

/***************************************************************************
Desc: 	Convert a context value into a storage text string
***************************************************************************/
RCODE F_Record::contextToText(
	FLMBYTE *	pucValue,
	FLMBYTE *	pucOutBuffer,
	FLMUINT *	puiBufLen)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiMaxOutLen;
	FLMUINT		uiStrLen;
	FLMUINT		uiBytesOutput = 0;
	FLMBOOL		bOutputData;
	FLMBYTE		ucTmpBuf[ 32];

	uiMaxOutLen = *puiBufLen;
	bOutputData = ((pucOutBuffer != NULL) && uiMaxOutLen) ? TRUE : FALSE;

	if( !pucValue)
	{
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "%u", FB2UD( pucValue));
	uiStrLen = f_strlen( ucTmpBuf) + 1;
	uiBytesOutput = f_min( uiStrLen, uiMaxOutLen);

	if( bOutputData)
	{
		f_memcpy( pucOutBuffer, ucTmpBuf, uiBytesOutput);
		pucOutBuffer[ uiBytesOutput - 1] = 0;
	}

	if( uiMaxOutLen < uiBytesOutput)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

Exit:

	*puiBufLen = uiBytesOutput;
	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getRootBlock(
	F_4x_LFILE *	pLFile,
	BTSK *			pStack)
{
	RCODE				rc = NE_XFLM_OK;
	F_Block *		pBlock = NULL;

	if( RC_BAD( rc = readBlock( pLFile->uiRootBlk, &pBlock)))
	{
		goto Exit;
	}

	if( !(FLM_4x_BH_IS_ROOT_BLK( pBlock->m_pucBlk)) ||
		(pLFile->uiLfNum != FB2UW( &pBlock->m_pucBlk[ 
			FLM_4x_BH_LOG_FILE_NUM])))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Set up the stack

	blkToStack( &pBlock, pStack);

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getBlock(
	F_4x_LFILE *	pLFile,
	FLMUINT			uiBlkAddr,
	BTSK *			pStack)
{
	RCODE				rc = NE_XFLM_OK;
	F_Block *		pBlock = NULL;

	if( RC_BAD( rc = readBlock( uiBlkAddr, &pBlock)))
	{
		goto Exit;
	}

	if( pLFile->uiLfNum != FB2UW( &pBlock->m_pucBlk[ 
			FLM_4x_BH_LOG_FILE_NUM]))
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	// Set up the stack

	blkToStack( &pBlock, pStack);

Exit:

	if( pBlock)
	{
		pBlock->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
void F_4xReader::blkToStack(
	F_Block **		ppBlock,
	BTSK *			pStack)
{
	FLMBYTE *		pucBlk = (*ppBlock)->m_pucBlk;
	FLMUINT			uiBlkType;

	uiBlkType = (FLMUINT)(FLM_4x_BH_GET_TYPE( pucBlk));
	pStack->uiBlkType = uiBlkType;

	if( uiBlkType == FLM_4x_BHT_LEAF)
	{
		pStack->uiElmOvhd = FLM_4x_BBE_KEY;
	}
	else if( uiBlkType == FLM_4x_BHT_NON_LEAF_DATA)
	{
		pStack->uiElmOvhd = FLM_4x_BNE_DATA_OVHD;
	}
	else if( uiBlkType == FLM_4x_BHT_NON_LEAF)
	{
		pStack->uiElmOvhd = FLM_4x_BNE_KEY_START;
	}
	else if( uiBlkType == FLM_4x_BHT_NON_LEAF_COUNTS)
	{
		pStack->uiElmOvhd = FLM_4x_BNE_KEY_COUNTS_START;
	}
	else
	{
		flmAssert( 0);
		pStack->uiElmOvhd = FLM_4x_BNE_KEY_START;
	}

	pStack->uiKeyLen = 0;
	pStack->uiPKC = 0;
	pStack->uiPrevElmPKC = 0;
	pStack->uiCurElm = FLM_4x_BH_OVHD;
	pStack->uiBlkEnd = (FLMUINT)FB2UW( &pucBlk[ FLM_4x_BH_ELM_END]);
	pStack->uiLevel = (FLMUINT)pucBlk[ FLM_4x_BH_LEVEL ];
	
	if( pStack->pBlk)
	{
		pStack->pBlk->Release();
	}

	pStack->pBlk = *ppBlock;
	*ppBlock = NULL;
}

/****************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getFieldType(
	FLMUINT		uiFieldNum,
	FLMUINT *	puiType)
{
	RCODE			rc = NE_XFLM_OK;

	if( uiFieldNum < m_uiFieldTblSize)
	{
		if( (*puiType = m_puiFieldTbl[ uiFieldNum - 1]) == 
			FLM_4x_UNKNOWN_TYPE)
		{
			rc = RC_SET( NE_XFLM_BAD_ELEMENT_NUM);
			goto Exit;
		}
	}
	else
	{
		// Check if the field is a FLAIM dictionary field.
		// Most of these fields are TEXT fields.

		if( (uiFieldNum >= FLM_4x_DICT_FIELD_NUMS) &&
			(uiFieldNum <= FLM_4x_LAST_DICT_FIELD_NUM))
		{
			*puiType = FLM_4x_TEXT_TYPE;
		}
		else if( uiFieldNum >= FLM_4x_UNREGISTERED_TAGS)
		{
			*puiType = FLM_4x_TEXT_TYPE;
		}
		else
		{
			rc = RC_SET( NE_XFLM_BAD_ELEMENT_NUM);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::readDictionary( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiType;
	FLMUINT		uiFieldNum;
	F_Record *	pRec = NULL;
	void *		pvField;

	if( m_puiFieldTbl)
	{
		f_free( &m_puiFieldTbl);
	}
	m_uiFieldTblSize = 0;

	if( RC_BAD( rc = f_alloc( FLM_4x_RESERVED_TAG_NUMS, 
		&m_puiFieldTbl)))
	{
		goto Exit;
	}

	m_uiFieldTblSize = FLM_4x_RESERVED_TAG_NUMS;
	f_memset( m_puiFieldTbl, FLM_4x_UNKNOWN_TYPE, m_uiFieldTblSize);
	setDefaultContainer( FLM_4x_DICT_CONTAINER);

	for( ;;)
	{
		if( RC_BAD( rc = retrieveNextRec( &pRec)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
			break;
		}

		if( pRec->getFieldID( pRec->root()) == FLM_4x_FIELD_TAG)
		{
			pvField = pRec->firstChild( pRec->root());
			if( RC_BAD( rc = getTypeTag( pRec, pvField, &uiType)))
			{
				goto Exit;
			}

			uiFieldNum = pRec->getID();
			flmAssert( uiFieldNum < FLM_4x_RESERVED_TAG_NUMS);
			m_puiFieldTbl[ uiFieldNum - 1] = uiType;
		}
	}

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		
****************************************************************************/
RCODE F_4xReader::getTypeTag(
	F_Record *		pRec,
	void *			pvField,
	FLMUINT *		puiType)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiType;
	FLMUINT		uiBufLen;
	char			szTmpBuf[ 64];

	uiBufLen = sizeof( szTmpBuf);
	if( RC_BAD( rc = pRec->getNative( pvField, szTmpBuf, &uiBufLen)))
	{
		goto Exit;
	}

	if( f_strnicmp( szTmpBuf, "text", 4) == 0)
	{
		uiType = FLM_4x_TEXT_TYPE;
	}
	else if( f_strnicmp( szTmpBuf, "numb", 4) == 0)
	{
		uiType = FLM_4x_NUMBER_TYPE;
	}
	else if( f_strnicmp( szTmpBuf, "bina", 4) == 0)
	{
		uiType = FLM_4x_BINARY_TYPE;
	}
	else if( f_strnicmp( szTmpBuf, "cont", 4) == 0)
	{
		uiType = FLM_4x_CONTEXT_TYPE;
	}
	else if( f_strnicmp( szTmpBuf, "blob", 4) == 0)
	{
		uiType = FLM_4x_BLOB_TYPE;
	}
	else
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	*puiType = uiType;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Constructor
****************************************************************************/
F_4xNameTable::F_4xNameTable()
{
	m_pool.poolInit( 1024);
	m_ppSortedByTagName = NULL;
	m_ppSortedByTagNum = NULL;
	m_ppSortedByTagTypeAndName = NULL;
	m_uiTblSize = 0;
	m_uiNumTags = 0;
	m_bTablesSorted = FALSE;
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_4xNameTable::~F_4xNameTable()
{
	clearTable();
	m_pool.poolFree();
}

/****************************************************************************
Desc:		Free everything in the table
****************************************************************************/
void F_4xNameTable::clearTable( void)
{
	m_pool.poolFree();
	m_pool.poolInit( 1024);

	// NOTE: Only one allocation is used for m_ppSortedByTagName,
	// m_ppSortedByTagNum, and m_ppSortedByTagTypeAndName - there is no
	// need to free m_ppSortedByTagNum and m_ppSortedByTagTypeAndName.

	if (m_ppSortedByTagName)
	{
		f_free( &m_ppSortedByTagName);
		m_ppSortedByTagNum = NULL;
		m_ppSortedByTagTypeAndName = NULL;
		m_uiTblSize = 0;
		m_uiNumTags = 0;
	}
}

/****************************************************************************
Desc:		Compare two tag names.  Name1 can be NATIVE or UNICODE.  If a
			non-NULL UNICODE string is passed, it will be used.  Otherwise,
			the NATIVE string will be used.
Note:		Comparison is case insensitive for the ASCII characters A-Z.
****************************************************************************/
FLMINT F_4xNameTable::tagNameCompare(
	const FLMUNICODE *	puzName1,	// If NULL, use pszName1 for comparison
	const char *			pszName1,
	const FLMUNICODE *	puzName2)
{
	FLMUNICODE	uzChar1;
	FLMUNICODE	uzChar2;

	if (puzName1)
	{
		for (;;)
		{
			uzChar1 = *puzName1;
			uzChar2 = *puzName2;

			// Convert to lower case for comparison.

			if (uzChar1 >= 'A' && uzChar1 <= 'Z')
			{
				uzChar1 = uzChar1 - 'A' + 'a';
			}
			if (uzChar2 >= 'A' && uzChar2 <= 'Z')
			{
				uzChar2 = uzChar2 - 'A' + 'a';
			}

			if (!uzChar1 || !uzChar2 || uzChar1 != uzChar2)
			{
				break;
			}

			puzName1++;
			puzName2++;
		}
	}
	else
	{
		for (;;)
		{
			uzChar1 = (FLMUNICODE)*pszName1;
			uzChar2 = *puzName2;

			// Convert to lower case for comparison.

			if (uzChar1 >= 'A' && uzChar1 <= 'Z')
			{
				uzChar1 = uzChar1 - 'A' + 'a';
			}
			if (uzChar2 >= 'A' && uzChar2 <= 'Z')
			{
				uzChar2 = uzChar2 - 'A' + 'a';
			}

			if (!uzChar1 || !uzChar2 || uzChar1 != uzChar2)
			{
				break;
			}

			pszName1++;
			puzName2++;
		}
	}

	if (uzChar1)
	{
		return( (FLMINT)((uzChar2 && uzChar1 < uzChar2)
								? (FLMINT)-1
								: (FLMINT)1));
	}
	else if (uzChar2)
	{
		return( -1);
	}
	else
	{
		return( 0);
	}
}

/****************************************************************************
Desc:		Lookup a tag by tag name.  Tag name is passed in as a UNICODE
			string or a NATIVE string.  If a non-NULL UNICODE string is
			passed in, it will be used.  Otherwise, the NATIVE string will
			be used.
****************************************************************************/
FLM_4x_TAG_INFO * F_4xNameTable::findTagByName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT *				puiInsertPos)
{
	FLM_4x_TAG_INFO *	pTagInfo = NULL;
	FLMUINT				uiTblSize;
	FLMUINT				uiLow;
	FLMUINT				uiMid;
	FLMUINT				uiHigh;
	FLMINT				iCmp;

	// Do binary search in the table

	if ((uiTblSize = m_uiNumTags) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}
	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;
		iCmp = tagNameCompare( puzTagName, pszTagName,
						m_ppSortedByTagName [uiMid]->puzTagName);
		if (iCmp == 0)
		{

			// Found Match
			
			pTagInfo = m_ppSortedByTagName [uiMid];
			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (iCmp < 0)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pTagInfo);
}

/****************************************************************************
Desc:		Lookup a tag by tag number.
****************************************************************************/
FLM_4x_TAG_INFO * F_4xNameTable::findTagByNum(
	FLMUINT		uiTagNum,
	FLMUINT *	puiInsertPos)
{
	FLM_4x_TAG_INFO *	pTagInfo = NULL;
	FLMUINT				uiTblSize;
	FLMUINT				uiLow;
	FLMUINT				uiMid;
	FLMUINT				uiHigh;
	FLMUINT				uiTblTagNum;

	// Do binary search in the table

	if ((uiTblSize = m_uiNumTags) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}

	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;
		uiTblTagNum = m_ppSortedByTagNum [uiMid]->uiTagNum;
		if (uiTagNum == uiTblTagNum)
		{

			// Found Match
			
			pTagInfo = m_ppSortedByTagNum [uiMid];
			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (uiTagNum < uiTblTagNum)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (uiTagNum < uiTblTagNum)
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pTagInfo);
}

/****************************************************************************
Desc:		Lookup a tag by tag type and tag name.  Tag name is passed
			in as a UNICODE string or a NATIVE string.  If a non-NULL
			UNICODE string is passed in, it will be used.  Otherwise, the
			NATIVE string will be used.
****************************************************************************/
FLM_4x_TAG_INFO * F_4xNameTable::findTagByTypeAndName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiType,
	FLMUINT *				puiInsertPos)
{
	FLM_4x_TAG_INFO *	pTagInfo = NULL;
	FLMUINT				uiTblType;
	FLMUINT				uiTblSize;
	FLMUINT				uiLow;
	FLMUINT				uiMid;
	FLMUINT				uiHigh;
	FLMINT				iCmp;

	// Do binary search in the table

	if ((uiTblSize = m_uiNumTags) == 0)
	{
		if (puiInsertPos)
		{
			*puiInsertPos = 0;
		}
		goto Exit;
	}
	uiHigh = --uiTblSize;
	uiLow = 0;
	for (;;)
	{
		uiMid = (uiLow + uiHigh) / 2;
		uiTblType = m_ppSortedByTagTypeAndName [uiMid]->uiType;
		if (uiType < uiTblType)
		{
			iCmp = -1;
		}
		else if (uiType > uiTblType)
		{
			iCmp = 1;
		}
		else if ((iCmp = tagNameCompare( puzTagName, pszTagName,
						m_ppSortedByTagTypeAndName [uiMid]->puzTagName)) == 0)
		{

			// Found Match
			
			pTagInfo = m_ppSortedByTagTypeAndName [uiMid];
			if (puiInsertPos)
			{
				*puiInsertPos = uiMid;
			}
			goto Exit;
		}

		// Check if we are done

		if (uiLow >= uiHigh)
		{

			// Done, item not found

			if (puiInsertPos)
			{
				*puiInsertPos = (iCmp < 0)
									 ? uiMid
									 : uiMid + 1;
			}
			goto Exit;
		}

		if (iCmp < 0)
		{
			if (uiMid == 0)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = 0;
				}
				goto Exit;
			}
			uiHigh = uiMid - 1;
		}
		else
		{
			if (uiMid == uiTblSize)
			{
				if (puiInsertPos)
				{
					*puiInsertPos = uiMid + 1;
				}
				goto Exit;
			}
			uiLow = uiMid + 1;
		}
	}

Exit:

	return( pTagInfo);
}

/****************************************************************************
Desc:		Copy a tag name to a UNICODE or NATIVE buffer.  Truncate if
			necessary.  If a non-NULL UNICODE string is passed in, it will
			be populated.  Otherwise, the NATIVE string will be populated.
****************************************************************************/
void F_4xNameTable::copyTagName(
	FLMUNICODE *	puzDestTagName,
	char *			pszDestTagName,
	FLMUINT			uiDestBufSize,	// Bytes, must be enough for null terminator
	FLMUNICODE *	puzSrcTagName)
{
	if (puzDestTagName)
	{

		// Decrement name buffer size by sizeof( FLMUNICODE) to allow for a
		// terminating NULL character. uiDestBufSize better be at list big
		// enough for a null terminating character.

		flmAssert( uiDestBufSize >= sizeof( FLMUNICODE));
		uiDestBufSize -= sizeof( FLMUNICODE);

		// Copy the name to the NATIVE buffer.  Non-Ascii UNICODE characters
		// will be returned as question marks (?).

		while (uiDestBufSize >= sizeof( FLMUNICODE) && *puzSrcTagName)
		{
			*puzDestTagName++ = *puzSrcTagName;
			uiDestBufSize -= sizeof( FLMUNICODE);
			puzSrcTagName++;
		}
		*puzDestTagName = 0;
	}
	else
	{
		// Decrement name buffer size by one to allow for a terminating
		// NULL character. uiDestBufSize better be at list big
		// enough for a null terminating character.

		flmAssert( uiDestBufSize);
		uiDestBufSize--;

		// Copy the name to the NATIVE buffer.  Non-Ascii UNICODE characters
		// will be returned as question marks (?).

		while (uiDestBufSize && *puzSrcTagName)
		{
			if (*puzSrcTagName <= 127)
			{
				*pszDestTagName++ = (char)*puzSrcTagName;
			}
			else
			{
				*pszDestTagName++ = '?';
			}
			uiDestBufSize--;
			puzSrcTagName++;
		}
		*pszDestTagName = 0;
	}
}

/***************************************************************************
Desc:		Sort an array of SCACHE pointers by their block address.
****************************************************************************/
void F_4xNameTable::sortTagTbl(
	FLM_4x_TAG_INFO **			ppTagInfoTbl,
	FLMUINT							uiLowerBounds,
	FLMUINT							uiUpperBounds,
	FLM_4x_TAG_COMPARE_FUNC		fnTagCompare)
{
	FLMUINT				uiLBPos;
	FLMUINT				uiUBPos;
	FLMUINT				uiMIDPos;
	FLMUINT				uiLeftItems;
	FLMUINT				uiRightItems;
	FLM_4x_TAG_INFO *	pCurTagInfo;
	FLMINT				iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurTagInfo = ppTagInfoTbl [uiMIDPos ];
	for (;;)
	{
		while (uiLBPos == uiMIDPos ||				// Don't compare with target
					((iCompare = 
						fnTagCompare( ppTagInfoTbl [uiLBPos], pCurTagInfo)) < 0))
		{
			if (uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while (uiUBPos == uiMIDPos ||				// Don't compare with target
					(((iCompare = 
						fnTagCompare( pCurTagInfo, ppTagInfoTbl [uiUBPos])) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}
		
		if (uiLBPos < uiUBPos )			// Interchange and continue loop.
		{

			// Exchange [uiLBPos] with [uiUBPos].

			tagInfoSwap( ppTagInfoTbl, uiLBPos, uiUBPos);
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else									// Past each other - done
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		tagInfoSwap( ppTagInfoTbl, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{

		// Exchange [uUBPos] with [uiMIDPos]

		tagInfoSwap( ppTagInfoTbl, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if( uiLeftItems < uiRightItems )
	{

		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if (uiLeftItems )
		{
			sortTagTbl( ppTagInfoTbl, uiLowerBounds, uiMIDPos - 1, fnTagCompare);
		}
		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if (uiLeftItems )	// Compute a truth table to figure out this check.
	{

		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			sortTagTbl( ppTagInfoTbl, uiMIDPos + 1, uiUpperBounds, fnTagCompare);
		}
		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/****************************************************************************
Desc:		Allocate a new tag info structure and set it up.
****************************************************************************/
RCODE F_4xNameTable::allocTag(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiTagNum,
	FLMUINT					uiType,
	FLMUINT					uiSubType,
	FLM_4x_TAG_INFO **	ppTagInfo)
{
	RCODE					rc = NE_XFLM_OK;
	void *				pvMark;
	FLM_4x_TAG_INFO *	pTagInfo;
	FLMUINT				uiNameSize;
	FLMUNICODE *		puzTmp;

	// Create a new tag info structure.

	pvMark = m_pool.poolMark();
	if( RC_BAD( rc = m_pool.poolAlloc( sizeof( FLM_4x_TAG_INFO), 
		(void **)&pTagInfo)))
	{
		goto Exit;
	}

	// Allocate the space for the tag name.

	if (puzTagName)
	{
		uiNameSize = (f_unilen( puzTagName) + 1) * sizeof( FLMUNICODE);
		if( RC_BAD( rc = m_pool.poolAlloc( uiNameSize, 
			(void **)&pTagInfo->puzTagName)))
		{
			goto Exit;
		}
		f_memcpy( pTagInfo->puzTagName, puzTagName, uiNameSize);
	}
	else
	{
		uiNameSize = (f_strlen( pszTagName) + 1) * sizeof( FLMUNICODE);
		if( RC_BAD( rc = m_pool.poolAlloc( uiNameSize, 
			(void **)&pTagInfo->puzTagName)))
		{
			goto Exit;
		}
		puzTmp = pTagInfo->puzTagName;
		while (*pszTagName)
		{
			*puzTmp++ = (FLMUNICODE)*pszTagName;
			pszTagName++;
		}
		*puzTmp = 0;
	}
	pTagInfo->uiTagNum = uiTagNum;
	pTagInfo->uiType = uiType;
	pTagInfo->uiSubType = uiSubType;

Exit:

	if (RC_BAD( rc))
	{
		m_pool.poolReset( pvMark);
		pTagInfo = NULL;
	}
	*ppTagInfo = pTagInfo;

	return( rc);
}

/****************************************************************************
Desc:		Allocate the sort tables.
****************************************************************************/
RCODE F_4xNameTable::reallocSortTables(
	FLMUINT	uiNewTblSize)
{
	RCODE						rc = NE_XFLM_OK;
	FLM_4x_TAG_INFO **	ppNewTbl;

	if( RC_BAD( rc = f_alloc( 
		sizeof( FLM_4x_TAG_INFO *) * uiNewTblSize * 3, &ppNewTbl)))
	{
		goto Exit;
	}

	// Copy the old tables into the new.

	if (m_uiNumTags)
	{
		f_memcpy( ppNewTbl, m_ppSortedByTagName,
						sizeof( FLM_4x_TAG_INFO *) * m_uiNumTags);
		f_memcpy( &ppNewTbl [uiNewTblSize], m_ppSortedByTagNum,
						sizeof( FLM_4x_TAG_INFO *) * m_uiNumTags);
		f_memcpy( &ppNewTbl [uiNewTblSize + uiNewTblSize],
						m_ppSortedByTagTypeAndName,
						sizeof( FLM_4x_TAG_INFO *) * m_uiNumTags);
		f_free( &m_ppSortedByTagName);
	}
	m_ppSortedByTagName = ppNewTbl;
	m_ppSortedByTagNum = &ppNewTbl [uiNewTblSize];
	m_ppSortedByTagTypeAndName = &ppNewTbl [uiNewTblSize + uiNewTblSize];
	m_uiTblSize = uiNewTblSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Get a tag name, number, etc. using tag number ordering.
			Tag name is returned as a UNICODE string or NATIVE string. If a
			non-NULL UNICODE string is passed in, it will be used.
			Otherwise, the NATIVE string will be used.
****************************************************************************/
FLMBOOL F_4xNameTable::getNextTagNumOrder(
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,			// May be NULL
	FLMUINT *		puiType,				// May be NULL
	FLMUINT *		puiSubType)			// May be NULL
{
	FLM_4x_TAG_INFO *		pTagInfo = NULL;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	if (*puiNextPos < m_uiNumTags)
	{
		pTagInfo = m_ppSortedByTagNum [*puiNextPos];
		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
								pTagInfo->puzTagName);
		}

		// Returned *puiNextPos should be the next one to retrieve.

		(*puiNextPos)++;
	}
	else
	{
		// Nothing more in list, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}

		if (pszTagName)
		{
			*pszTagName = 0;
		}

		if (puiTagNum)
		{
			*puiTagNum = 0;
		}

		if (puiType)
		{
			*puiType = 0;
		}

		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:		Get a tag name, number, etc. using tag name ordering.
			Tag name is returned as a UNICODE string or NATIVE string. If a
			non-NULL UNICODE string is passed in, it will be used.
			Otherwise, the NATIVE string will be used.
****************************************************************************/
FLMBOOL F_4xNameTable::getNextTagNameOrder(
	FLMUINT *		puiNextPos,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,
	FLMUINT *		puiTagNum,			// May be NULL
	FLMUINT *		puiType,				// May be NULL
	FLMUINT *		puiSubType)			// May be NULL
{
	FLM_4x_TAG_INFO *		pTagInfo = NULL;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	if (*puiNextPos < m_uiNumTags)
	{
		pTagInfo = m_ppSortedByTagName [*puiNextPos];
		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
								pTagInfo->puzTagName);
		}

		// Returned *puiNextPos should be the next one to retrieve.

		(*puiNextPos)++;
	}
	else
	{
		// Nothing more in list, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}

		if (pszTagName)
		{
			*pszTagName = 0;
		}

		if (puiTagNum)
		{
			*puiTagNum = 0;
		}

		if (puiType)
		{
			*puiType = 0;
		}

		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:		Get a tag name and number from type.  Tag name is returned as a
			UNICODE string or NATIVE string. If a non-NULL UNICODE string is
			passed in, it will be used.  Otherwise, the NATIVE string
			will be used.
****************************************************************************/
FLMBOOL F_4xNameTable::getFromTagType(
	FLMUINT			uiType,
	FLMUINT *		puiNextPos,		// To get first, initialize to zero.
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,	// In bytes - must be at least sizeof( FLMUNICODE)
	FLMUINT *		puiTagNum,		// May be NULL
	FLMUINT *		puiSubType)		// May be NULL
{
	FLM_4x_TAG_INFO *		pTagInfo = NULL;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	if (*puiNextPos == 0)
	{

		// A value of zero indicates we should try to find the first
		// one.

		(void)findTagByTypeAndName( NULL, "", uiType, puiNextPos);
		if (*puiNextPos < m_uiNumTags &&
			 m_ppSortedByTagTypeAndName [*puiNextPos]->uiType != uiType)
		{
			(*puiNextPos)++;
		}
	}

	if (*puiNextPos < m_uiNumTags &&
		 m_ppSortedByTagTypeAndName [*puiNextPos]->uiType == uiType)
	{
		pTagInfo = m_ppSortedByTagTypeAndName [*puiNextPos];

		if (puiTagNum)
		{
			*puiTagNum = pTagInfo->uiTagNum;
		}
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
								pTagInfo->puzTagName);
		}

		// Returned *puiNextPos should be the next one to retrieve, so that
		// it is not zero.

		(*puiNextPos)++;
	}
	else
	{
		// Type was not found, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}
		if (pszTagName)
		{
			*pszTagName = 0;
		}
		if (puiTagNum)
		{
			*puiTagNum = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:		Get a tag name from its tag number.  Tag name is returned as a
			UNICODE string or NATIVE string. If a non-NULL UNICODE string
			is passed in, it will be used.  Otherwise, the NATIVE string
			will be used.
****************************************************************************/
FLMBOOL F_4xNameTable::getFromTagNum(
	FLMUINT			uiTagNum,
	FLMUNICODE *	puzTagName,
	char *			pszTagName,
	FLMUINT			uiNameBufSize,	// In bytes, at least enough for null char.
	FLMUINT *		puiType,			// May be NULL
	FLMUINT *		puiSubType)		// May be NULL
{
	FLM_4x_TAG_INFO *		pTagInfo;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	if ((pTagInfo = findTagByNum( uiTagNum)) != NULL)
	{
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}

		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}

		if( puzTagName || pszTagName)
		{
			copyTagName( puzTagName, pszTagName, uiNameBufSize,
							pTagInfo->puzTagName);
		}
	}
	else
	{
		// Tag number was not found, but initialize return variables anyway.

		if (puzTagName)
		{
			*puzTagName = 0;
		}

		if (pszTagName)
		{
			*pszTagName = 0;
		}

		if (puiType)
		{
			*puiType = 0;
		}

		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:		Get a tag number and type from its tag name.  Tag name is passed
			in as a UNICODE string or NATIVE string. If a non-NULL UNICODE
			string is passed in, it will be used.  Otherwise, the NATIVE
			string will be used.
****************************************************************************/
FLMBOOL F_4xNameTable::getFromTagName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT *				puiTagNum,		// Cannot be NULL
	FLMUINT *				puiType,			// May be NULL
	FLMUINT *				puiSubType)		// May be NULL
{
	FLM_4x_TAG_INFO *		pTagInfo;

	if (!m_bTablesSorted)
	{
		sortTags();
	}

	if ((pTagInfo = findTagByName( puzTagName, pszTagName)) != NULL)
	{
		flmAssert( puiTagNum);
		*puiTagNum = pTagInfo->uiTagNum;
		if (puiType)
		{
			*puiType = pTagInfo->uiType;
		}

		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}
	}
	else
	{
		// Tag name was not found, but initialize return variables anyway.

		*puiTagNum = 0;
		if (puiType)
		{
			*puiType = 0;
		}
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:		Get a tag number from its tag name and type.  Tag name is passed
			in as a UNICODE or NATIVE string. If a non-NULL UNICODE string is
			passed in, it will be used.  Otherwise, the NATIVE string will
			be used.
****************************************************************************/
FLMBOOL F_4xNameTable::getFromTagTypeAndName(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiType,
	FLMUINT *				puiTagNum,	// Cannot be NULL
	FLMUINT *				puiSubType)	// May be NULL
{
	FLM_4x_TAG_INFO *		pTagInfo;

	if (!m_bTablesSorted)
	{
		sortTags();
	}
	
	if ((pTagInfo = findTagByTypeAndName( puzTagName, pszTagName,
									uiType)) != NULL)
	{
		flmAssert( puiTagNum);
		*puiTagNum = pTagInfo->uiTagNum;
		if (puiSubType)
		{
			*puiSubType = pTagInfo->uiSubType;
		}
	}
	else
	{

		// Tag name was not found, but initialize return variables anyway.

		*puiTagNum = 0;
		if (puiSubType)
		{
			*puiSubType = 0;
		}
	}

	return( (FLMBOOL)(pTagInfo ? (FLMBOOL)TRUE : (FLMBOOL)FALSE));
}

/****************************************************************************
Desc:		Insert a tag info structure into the sorted tables at the
			specified positions.
****************************************************************************/
RCODE F_4xNameTable::insertTagInTables(
	FLM_4x_TAG_INFO *	pTagInfo,
	FLMUINT				uiTagNameTblInsertPos,
	FLMUINT				uiTagTypeAndNameTblInsertPos,
	FLMUINT				uiTagNumTblInsertPos)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLoop;

	// See if we need to resize the tables.  Start at 256.  Double each
	// time up to 2048.  Then just add 2048 at a time.

	if (m_uiNumTags == m_uiTblSize)
	{
		FLMUINT	uiNewSize;

		if (!m_uiTblSize)
		{
			uiNewSize = 256;
		}
		else if (m_uiTblSize < 2048)
		{
			uiNewSize = m_uiTblSize * 2;
		}
		else
		{
			uiNewSize = m_uiTblSize + 2048;
		}

		if (RC_BAD( rc = reallocSortTables( uiNewSize)))
		{
			goto Exit;
		}
	}

	// Insert into the sorted-by-name table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagNameTblInsertPos)
	{
		m_ppSortedByTagName [uiLoop] = m_ppSortedByTagName [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagName [uiTagNameTblInsertPos] = pTagInfo;

	// Insert into the sorted-by-number table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagNumTblInsertPos)
	{
		m_ppSortedByTagNum [uiLoop] = m_ppSortedByTagNum [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagNum [uiTagNumTblInsertPos] = pTagInfo;

	// Insert into the sorted-by-tag-name-and-type table

	uiLoop = m_uiNumTags;
	while (uiLoop > uiTagTypeAndNameTblInsertPos)
	{
		m_ppSortedByTagTypeAndName [uiLoop] =
			m_ppSortedByTagTypeAndName [uiLoop - 1];
		uiLoop--;
	}
	m_ppSortedByTagTypeAndName [uiTagTypeAndNameTblInsertPos] = pTagInfo;

	// Increment the total number of tags

	m_uiNumTags++;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Add a tag to the table.  Tag name is passed in as a UNICODE
			string or NATIVE string. If a non-NULL UNICODE string is passed
			in, it will be used.  Otherwise, the NATIVE string will be used.
****************************************************************************/
RCODE F_4xNameTable::addTag(
	const FLMUNICODE *	puzTagName,
	const char *			pszTagName,
	FLMUINT					uiTagNum,
	FLMUINT					uiType,
	FLMUINT					uiSubType,
	FLMBOOL					bCheckDuplicates)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiTagNameTblInsertPos;
	FLMUINT				uiTagTypeAndNameTblInsertPos;
	FLMUINT				uiTagNumTblInsertPos;
	FLM_4x_TAG_INFO *	pTagInfo;

	// Must have a non-NULL tag name.  Use UNICODE string if it is
	// non-NULL.  Otherwise, use NATIVE string.

	if (puzTagName && *puzTagName)
	{
		pszTagName = NULL;
	}
	else if (pszTagName && *pszTagName)
	{
		puzTagName = NULL;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	// Tag number of zero not allowed.

	if (!uiTagNum)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	// Tables must be sorted in order for this to work

	if (bCheckDuplicates)
	{
		if (!m_bTablesSorted)
		{
			sortTags();
		}

		// Make sure that the tag name is not already used.

		if (findTagByName( puzTagName, pszTagName, &uiTagNameTblInsertPos))
		{
			rc = RC_SET( NE_XFLM_EXISTS);
			goto Exit;
		}

		// Make sure that the tag name + type is not already used.

		if (findTagByTypeAndName( puzTagName, pszTagName,
						uiType, &uiTagTypeAndNameTblInsertPos))
		{
			rc = RC_SET( NE_XFLM_EXISTS);
			goto Exit;
		}

		// Make sure that the tag number is not already used.

		if (findTagByNum( uiTagNum, &uiTagNumTblInsertPos))
		{
			rc = RC_SET( NE_XFLM_EXISTS);
			goto Exit;
		}
	}
	else
	{
		uiTagNameTblInsertPos =
		uiTagTypeAndNameTblInsertPos =
		uiTagNumTblInsertPos = m_uiNumTags;
		m_bTablesSorted = FALSE;
	}

	// Create a new tag info structure.

	if (RC_BAD( rc = allocTag( puzTagName, pszTagName, uiTagNum, uiType,
								uiSubType, &pTagInfo)))
	{
		goto Exit;
	}

	// Insert the tag structure into the appropriate places in the
	// sorted tables.

	if (RC_BAD( rc = insertTagInTables( pTagInfo, uiTagNameTblInsertPos,
							uiTagTypeAndNameTblInsertPos,
							uiTagNumTblInsertPos)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sort the tag tables according to their respective criteria.
****************************************************************************/
void F_4xNameTable::sortTags( void)
{
	if (!m_bTablesSorted && m_uiNumTags > 1)
	{
		sortTagTbl( m_ppSortedByTagName, 0, m_uiNumTags - 1,
				compareTagNameOnly);
		sortTagTbl( m_ppSortedByTagNum, 0, m_uiNumTags - 1,
				compareTagNumOnly);
		sortTagTbl( m_ppSortedByTagTypeAndName, 0, m_uiNumTags - 1,
				compareTagTypeAndName);
	}
	m_bTablesSorted = TRUE;
}

/****************************************************************************
Desc:		Initialize a name table from a database.
****************************************************************************/
RCODE F_4xNameTable::setupNameTable(
	F_4xReader *		pDb)
{
	RCODE				rc = NE_XFLM_OK;
	F_Record *		pRec = NULL;
	FLMUINT			uiDrn;
	FLMUINT			uiLoop;
	void *			pvField;
	FLMUNICODE		uzName[ 60];
	FLMUNICODE *	puzName = &uzName[ 0];
	FLMUINT			uiNameLen = sizeof( uzName);
	FLMUINT			uiLen;
	FLMUINT			uiSubType;

	// Clean out all existing tags, if any.

	clearTable();

	// Find the next DRN in the dictionary container

	if( RC_BAD( rc = pDb->getNextDrn( FLM_4x_DICT_CONTAINER, &uiDrn)))
	{
		goto Exit;
	}

	// Count the reserved tags

	for (uiLoop = 0; Flm4xDictTagInfo[ uiLoop].pszTagName; uiLoop++)
	{
		;
	}

	// Preallocate space so we don't have to do it over and over.

	if( RC_BAD( rc = reallocSortTables( uiLoop + uiDrn)))
	{
		goto Exit;
	}

	// Add in all of the reserved dictionary tags.

	for( uiLoop = 0; Flm4xDictTagInfo[ uiLoop].pszTagName; uiLoop++)
	{
		if( RC_BAD( rc = addTag( NULL,
								Flm4xDictTagInfo[ uiLoop].pszTagName,
								Flm4xDictTagInfo[ uiLoop].uiTagNum,
								FLM_4x_FIELD_TAG,
								Flm4xDictTagInfo[ uiLoop].uiFieldType, FALSE)))
		{
			goto Exit;
		}
	}

	// Read through all of the dictionary records

	pDb->setDefaultContainer( FLM_4x_DICT_CONTAINER);
	for( ;;)
	{
		if( RC_BAD( rc = pDb->retrieveNextRec( &pRec)))
		{
			if( rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}

			rc = NE_XFLM_OK;
			break;
		}

		pvField = pRec->root();
		uiDrn = pRec->getID();

		// Get the unicode name length (does not include NULL terminator)

		if( RC_BAD( rc = pRec->getUnicode( pvField, NULL, &uiLen)))
		{
			goto Exit;
		}

		// Account for NULL character.

		uiLen += sizeof( FLMUNICODE);

		// See if we need a larger buffer to get the name.

		if (uiLen > uiNameLen)
		{
			FLMUNICODE *	puzTmp;

			// Add enough for 60 more unicode characters.

			uiLen += (60 * sizeof( FLMUNICODE));

			if( RC_BAD( rc = f_alloc( uiLen, &puzTmp)))
			{
				goto Exit;
			}

			if (puzName != &uzName [0])
			{
				f_free( &puzName);
			}

			puzName = puzTmp;
			uiNameLen = uiLen;
		}

		// Get the tag name.

		uiLen = uiNameLen;
		if (RC_BAD( rc = pRec->getUnicode( pvField, puzName, &uiLen)))
		{
			goto Exit;
		}

		// Get the sub-type.

		if (pRec->getFieldID( pvField) == FLM_4x_FIELD_TAG)
		{
			void *	pvFld = pRec->find( pvField, FLM_4x_TYPE_TAG, 1, FALSE);

			if (!pvFld ||
				 RC_BAD( pDb->getTypeTag( pRec, pvFld, &uiSubType)))
			{
				uiSubType = FLM_4x_TEXT_TYPE;
			}
		}
		else
		{
			uiSubType = 0;
		}

		// Add tag to table, without sorting yet.

		if (RC_BAD( rc = addTag( puzName, NULL, uiDrn,
								pRec->getFieldID( pvField), uiSubType, FALSE)))
		{
			goto Exit;
		}
	}

	sortTags();

Exit:

	if( RC_BAD( rc))
	{
		clearTable();
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( puzName != &uzName [0])
	{
		f_free( &puzName);
	}

	return( rc);
}
