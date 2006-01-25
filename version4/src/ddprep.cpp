//-------------------------------------------------------------------------
// Desc:	Routines to verify all dictionary syntax.
// Tabs:	3
//
//		Copyright (c) 1992-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ddprep.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define FDD_MAX_VALUE_SIZE 		64

FSTATIC RCODE fdictAddDictIndex(
	TDICT *	 		pTDict);

FSTATIC RCODE DDFieldParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum);
	
FSTATIC RCODE DDGetReference(
	FlmRecord *		pRecord,	
	void *			pvField,
	const char *	pszBuffer,
	FLMUINT *		puiIdRef);
	
FSTATIC RCODE DDAllocEntry(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum,
	DDENTRY **		ppDDEntryRV);

FSTATIC RCODE DDIxParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord,
	void *			pvField);
	
FSTATIC RCODE DDBuildFldPath(
	TDICT *			pTDict,
	TIFD **			ppTIfd,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiBaseNum);
	
FSTATIC FLMBOOL DDMoveWord(
	char *			pucDest,
	char *			pucSrc,
	FLMUINT			uiMaxDestLen,
	FLMUINT *		puiPos);
	
FSTATIC RCODE DDContainerParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord);

FSTATIC void DDTextToNative(
	FlmRecord *		pRecord,
	void *			pvField,
	char *			pszBuffer,
	FLMUINT			uiBufLen,
	FLMUINT *		puiBufLen);

FSTATIC RCODE DDParseStateOptions(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT *		puiFldInfo);

FSTATIC RCODE	DDEncDefParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum);

FSTATIC RCODE DDGetEncKey(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	void *			pvField,
	TENCDEF *		pTEncDef);

#define MAX_ENC_TYPES	3

// NOTE:  If you change the arrangement of the values in this array, make sure
// you search the entire codebase for references to DDEncOpts and DDGetEncType
// and verify that the changes won't cause problems.  This is particularly
// important because these values DO NOT match up exactly with the values in
// the SMEncryptionScheme enum that's used at the SMI level.

char * DDEncOpts[ MAX_ENC_TYPES] =
{
	"aes",
	"des3",
	"des"
};

#define	START_DD_INDEX_OPTS  0
#define	DD_IX_FIELD_OPT		0
#define	DD_IX_COMPOUND_OPT	1
#define	DD_IX_UPPER_OPT		2
#define	DD_IX_EACHWORD_OPT	3
#define	DD_IX_MIXED_OPT		4
#define	DD_IX_CONTEXT_OPT		5
#define	DD_IX_POST_OPT			6
#define	MAX_DD_INDEX_OPTS    7

/****************************************************************************
Desc:  	Read all data dictionary records parsing and sending to process.
			All temporary structures are off of pTDict.  pTDict must be setup.
****************************************************************************/
RCODE fdictProcessAllDictRecs(
	FDB *			pDb,
	TDICT *		pTDict)
{
	RCODE			rc;
	LFILE *		pLFile = pTDict->pLFile;
	BTSK			stackBuf[ BH_MAX_LEVELS ];	// Stack to hold b-tree variables
	BTSK *		stack = stackBuf;		 		// Points to proper stack frame
	FLMBYTE		btKeyBuf[ DRN_KEY_SIZ +8];	// Key buffer pointed to by stack
	FLMBYTE		key[4];					 		// Used for dummy empty key
	FLMUINT		uiDrn;
	FlmRecord *	pRecord = NULL;

	// Add the dictionary index to the front of TDICT.
	if( RC_BAD( rc = fdictAddDictIndex( pTDict)))
	{
		goto Exit;
	}

	// Position to the first of the data dictionary data records & read.
	FSInitStackCache( &stackBuf [0], BH_MAX_LEVELS);
	stack->pKeyBuf = btKeyBuf;
	longToByte( 0, key);
	if( RC_BAD(rc = FSBtSearch( pDb, pLFile, &stack, key, DRN_KEY_SIZ, 0 )))
		goto Exit;

	// Special case of no records.
	if( stack->uiCmpStatus == BT_END_OF_DATA)	
		goto Exit;
	stack->uiFlags = NO_STACK;					// Fake out the stack for speed.

	do
	{
		uiDrn = (FLMUINT) byteToLong( btKeyBuf);
		if( uiDrn == DRN_LAST_MARKER)
		{
			break;
		}
		
		// VERY IMPORTANT NOTE:
		//  	DO NOT READ FROM CACHE - THE RECORD MAY
		// 	NOT HAVE BEEN PUT INTO RECORD CACHE YET, AND WE NEED TO HAVE
		// 	THE CORRECT VERSION OF THE RECORD.

		if( RC_BAD( rc = FSReadElement( pDb, &pDb->TempPool, pLFile, 
			uiDrn, stack, TRUE, &pRecord, NULL, NULL)))
		{
			break;
		}

		if( RC_BAD(rc = fdictProcessRec( pTDict, pRecord, uiDrn)))
		{
			pDb->Diag.uiDrn = uiDrn;
			pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
			if( pTDict->uiBadField != 0)
			{
				pDb->Diag.uiFieldNum = pTDict->uiBadField;
				pDb->Diag.uiInfoFlags |= FLM_DIAG_FIELD_NUM;
			}
			break;
		}

		// Position to the next record - SUCCESS or FERR_BT_END_OF_DATA
		rc = FSNextRecord( pDb, pLFile, stack);

	} while( RC_OK(rc));

	rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc );
}

/****************************************************************************
Desc:		Add the dictionary index to pTDict.
****************************************************************************/
RCODE fdictAddDictIndex(
	TDICT *	 		pTDict)
{
	RCODE				rc;
	DDENTRY *		pDDEntry;
	TIXD *			pTIxd;
	TIFD *			pTIfd;
	TIFP *			pTIfp;

	if( RC_BAD( rc = DDAllocEntry( pTDict, NULL, FLM_DICT_INDEX, &pDDEntry)))
	{
		goto Exit;
	}
	pDDEntry->uiType = ITT_INDEX_TYPE;

	if( (pTIxd = (TIXD *) GedPoolAlloc( &pTDict->pool, sizeof( TIXD))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pTDict->uiNewIxds++;
	pDDEntry->vpDef = (void *) pTIxd;
	pTIxd->uiFlags = IXD_UNIQUE;
	pTIxd->uiContainerNum = FLM_DICT_CONTAINER;
	pTIxd->uiNumFlds = 1;
	pTIxd->uiLanguage = pTDict->uiDefaultLanguage;
	pTIxd->uiEncId = 0;

	if( (pTIfd = (TIFD *) GedPoolAlloc( &pTDict->pool,	sizeof( TIFD))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pTIxd->pNextTIfd = pTIfd;
	pTDict->uiNewIfds++;
	pTIfd->pTIfp = NULL;
	pTIfd->pNextTIfd = NULL;
	pTIfd->uiFlags = (FLMUINT)(IFD_FIELD | FLM_TEXT_TYPE);
	pTIfd->uiNextFixupPos = 0;
	pTIfd->uiLimit = IFD_DEFAULT_LIMIT;
	pTIfd->uiCompoundPos = 0;

	if( (pTIfp = (TIFP *) GedPoolAlloc( &pTDict->pool, sizeof( TIFP ))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pTDict->uiNewFldPaths += 2;
	pTIfd->pTIfp = pTIfp;

	pTIfp->pNextTIfp = NULL;
	pTIfp->bFieldInThisDict = FALSE;
	pTIfp->uiFldNum = FLM_NAME_TAG;

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Process a single data dictionary record. Parse the record for syntax
			errors depending on flag value.  Only supports adding new stuff
			to pTDict.
****************************************************************************/
RCODE fdictProcessRec(
	TDICT *	 		pTDict,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum)
{
	RCODE      		rc = FERR_OK;
	DDENTRY *		pDDEntry;
	void *			pvField = pRecord->root();

	// Ignore items with root nodes that are in the unregistered range.

	if( pRecord->getFieldID( pvField) >= FLM_UNREGISTERED_TAGS)
	{
		goto Exit;
	}

	// Parse only on modify or add.

	switch( pRecord->getFieldID( pvField))
	{
		case FLM_FIELD_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry( 
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}

			pDDEntry->uiType = 0;				// Type of zero means field.
			if( RC_BAD( rc = DDFieldParse( pTDict, pDDEntry, 
							pRecord, uiDictRecNum)))
			{
				goto Exit;
			}
			break;
		}

		case FLM_INDEX_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry( 
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}
			pDDEntry->uiType = ITT_INDEX_TYPE;
			if( RC_BAD( rc = DDIxParse( pTDict, pDDEntry, pRecord, pvField)))
			{
				goto Exit;
			}
			pTDict->uiNewIxds++;
			break;
		}

		case FLM_CONTAINER_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry( 
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}
			pDDEntry->uiType = ITT_CONTAINER_TYPE;
			if( RC_BAD( rc = DDContainerParse( pTDict, pDDEntry, pRecord)))
			{
				goto Exit;
			}
			pTDict->uiTotalLFiles++;
			break;
		}
		case FLM_ENCDEF_TAG:
		{
			if( RC_BAD( rc = DDAllocEntry(
					pTDict, pRecord, uiDictRecNum, &pDDEntry)))
			{
				goto Exit;
			}
			pDDEntry->uiType = ITT_ENCDEF_TYPE;
			if (RC_BAD( rc = DDEncDefParse( pTDict, pDDEntry, pRecord, uiDictRecNum)))
			{
				goto Exit;
			}
			break;
		}
		case FLM_AREA_TAG:
		case FLM_RESERVED_TAG:
		{
			break;
		}

		default:
		{
			// Cannot allow anything else to pass through the dictionary.
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Allocate, check and add a name to the DDEntry structure.
****************************************************************************/
FSTATIC RCODE DDAllocEntry(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	FLMUINT			uiDictRecNum, 
	DDENTRY **		ppDDEntryRV)
{
	RCODE				rc = FERR_OK;
	DDENTRY *		pNewEntry;

	pNewEntry = (DDENTRY *)GedPoolAlloc( &pTDict->pool, sizeof(DDENTRY));
	if( !pNewEntry)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pNewEntry->pNextEntry = NULL;
	pNewEntry->vpDef = NULL;
	pNewEntry->uiEntryNum = uiDictRecNum;
	pNewEntry->uiType = 0;

	// Zero length name NOT allowed for dictionary items.

	if( pRecord)
	{
		if( pRecord->getDataLength( pRecord->root()) == 0)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( pTDict->pLastEntry)
	{
		pTDict->pLastEntry->pNextEntry = pNewEntry;
	}
	else
	{
		pTDict->pFirstEntry = pNewEntry;
	}
	pTDict->pLastEntry = pNewEntry;
	*ppDDEntryRV = pNewEntry;

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Parse field definition
****************************************************************************/
FSTATIC RCODE DDFieldParse(
	TDICT *		pTDict,
	DDENTRY *	pDDEntry,
	FlmRecord *	pRecord,
	FLMUINT		uiDictRecNum)
{
	RCODE    	rc = FERR_OK;
	TFIELD  *	pTField;
	void *		pvField;

	if( (pTField = (TFIELD *)GedPoolAlloc( &pTDict->pool, sizeof(TFIELD))) == NULL)
	{
		return( RC_SET( FERR_MEM));
	}

	pTField->uiFldNum = uiDictRecNum;
	pTField->uiFldInfo = FLM_CONTEXT_TYPE;
	pDDEntry->vpDef = (void *) pTField;

	if( (pvField = pRecord->firstChild( pRecord->root())) == NULL)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ; pvField; pvField = pRecord->nextSibling( pvField))
	{
		switch( pRecord->getFieldID( pvField))
		{
			case FLM_TYPE_TAG:
			{
				rc = DDGetFieldType( pRecord, pvField, &pTField->uiFldInfo);
				break;
			}

			case FLM_STATE_TAG:
			{
				rc = DDParseStateOptions( pRecord, pvField, &pTField->uiFldInfo);
				break;
			}

			default:
			{
				if( pRecord->getFieldID( pvField) < FLM_UNREGISTERED_TAGS &&
					 pRecord->getFieldID( pvField) != FLM_COMMENT_TAG)
				{
					rc = RC_SET( FERR_SYNTAX);
				}
				break;
			}
		}
	}

Exit:

	if( RC_BAD(rc) && pvField)
	{
		pTDict->uiBadField = pRecord->getFieldID( pvField);
	}
	return( rc );
}

/****************************************************************************
Desc:		Returns the fields data type.  May be called outside of DDPREP.C
****************************************************************************/
RCODE DDGetFieldType(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT *		puiFldInfo)
{
	RCODE				rc = FERR_OK;
	char				szNativeBuf[ FDD_MAX_VALUE_SIZE];	

	DDTextToNative( pRecord, pvField, szNativeBuf, FDD_MAX_VALUE_SIZE, NULL );

	// Parse the type keyword - only one type allowed.

	if (f_strnicmp( szNativeBuf, "text", 4) == 0)
	{
		*puiFldInfo = FLM_TEXT_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "numb", 4) == 0)
	{
		*puiFldInfo = FLM_NUMBER_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "bina", 4) == 0)
	{
		*puiFldInfo = FLM_BINARY_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "cont", 4) == 0)
	{
		*puiFldInfo = FLM_CONTEXT_TYPE;
	}
	else if (f_strnicmp( szNativeBuf, "blob", 4) == 0)
	{
		*puiFldInfo = FLM_BLOB_TYPE;
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
Desc:		Parses the 'state' option that is found within the 'field'
			dictionary definition.
Format: 	state [checking | unused | purge | active]
****************************************************************************/
FSTATIC RCODE DDParseStateOptions(
	FlmRecord *	pRecord,
	void *		pvField,
	FLMUINT *	puiFldInfo)
{
	RCODE			rc = FERR_OK;
	char			szNativeBuf[ FDD_MAX_VALUE_SIZE];

	DDTextToNative( pRecord, pvField, szNativeBuf, FDD_MAX_VALUE_SIZE, NULL);

	// Parse the 'state' keyword - only one type allowed

	if( f_strnicmp( szNativeBuf, "chec", 4) == 0)
	{
		// 0xFFCF is used to clear out any existing field 'state' value
		
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_CHECKING);
	}
	else if( f_strnicmp( szNativeBuf, "unus", 4) == 0)
	{
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_UNUSED);
	}
	else if( f_strnicmp( szNativeBuf, "purg", 4) == 0)
	{
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_PURGE);
	}
	else if( f_strnicmp( szNativeBuf, "acti", 4) == 0)
	{
		*puiFldInfo = (FLMUINT)((*puiFldInfo & ~ITT_FLD_STATE_MASK) | ITT_FLD_STATE_ACTIVE);
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
	}

	return( rc);
}

/****************************************************************************
Desc:		Get a number reference and set in the (OUT) parameter.
****************************************************************************/
FSTATIC RCODE DDGetReference(
	FlmRecord *		pRecord,
	void *			pvField,
	const char *	pszBuffer,
	FLMUINT *		puiIdRef)
{
	RCODE			rc = FERR_OK;

	*puiIdRef = 0;
	if( pszBuffer)
	{
		if( !(*pszBuffer))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		*puiIdRef = f_atoud( pszBuffer);
	}
	else
	{
		if( RC_BAD( rc = pRecord->getUINT( pvField, puiIdRef))) 
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}


/****************************************************************************
Desc:		Returns the encryption type.  May be called outside of DDPREP.C
****************************************************************************/
RCODE DDGetEncType(
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT *		puiFldInfo)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiType;
	char				szNativeBuf[ FDD_MAX_VALUE_SIZE];	

	DDTextToNative( pRecord, pvField, szNativeBuf, FDD_MAX_VALUE_SIZE, NULL );

	// Parse the type keyword - only one type allowed.

	for( uiType = 0;
		  uiType < MAX_ENC_TYPES ;
		  uiType++)
	{
		if( f_strnicmp( szNativeBuf, DDEncOpts[ uiType],
					f_strlen(DDEncOpts[ uiType])) == 0)
		{
			*puiFldInfo = uiType;
			goto Exit;
		}
	}

	rc = RC_SET( FERR_SYNTAX);

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Returns the binary key info.  May be called outside of DDPREP.C
****************************************************************************/
FSTATIC RCODE DDGetEncKey(
	TDICT *			pTDict,
	FlmRecord *		pRecord,
	void *			pvField,
	TENCDEF *		pTEncDef)
{
	RCODE				rc = FERR_OK;
	char *			pucBuffer = NULL;
	FLMUINT			uiLength;

	pTEncDef->uiLength = 0;

	if (RC_BAD( rc = pRecord->getNativeLength( pvField, &uiLength)))
	{
		goto Exit;
	}
	uiLength++;

	// Allocate the buffer from the pool so it will be easily freed later.
	
	if( (pucBuffer = (char *)GedPoolAlloc( &pTDict->pool, uiLength)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pRecord->getNative( pvField, pucBuffer, &uiLength)))
	{
		goto Exit;
	}

	pTEncDef->uiLength = uiLength;
	pTEncDef->pucKeyInfo = (FLMBYTE *)pucBuffer;

Exit:

	return( rc);

}

/****************************************************************************
Desc:		Parse an data dictionary index definition for correct syntax &
			assign the correct attributes.  Build the pcode buffer for the index.
Return:	RCODE - SUCCESS or FERR_SYNTAX
Format:

0 index <psName>						# FLM_INDEX_TAG
[ 1 area [ 0 | <ID>]]				# FLM_AREA_TAG - QF files area, 0 = "same as DB"
[ 1 container {DEFAULT | <ID>}]	# FLM_CONTAINER_TAG - indexes span only one container
[ 1 count [ KEYS &| REFS]]			# FLM_COUNT_TAG - key count of keys and/or refs
[ 1 language {US | <language>}]	# FLM_LANGUAGE_TAG - for full-text parsing and/or sorting
[ 1 positioning]						# FLM_POSITIONING_TAG - full reference counts at all b-tree elements
[ 1 encdef <EncryptionDefId>]		# FLM_ENCDEF_TAG - identify the encryption definition to use

  1 key [EACHWORD]					# FLM_KEY_TAG - 'use' defaults based on type
  [ 2	base <ID>]						# FLM_BASE_TAG - base rec/field for fields below
  [ 2 combinations <below> 	 	# FLM_COMBINATIONS_TAG - how to handle repeating fields
		{ ALL | NORMALIZED}]
  [ 2 post]								# FLM_POST_TAG - case-flags post-pended to key
  [ 2	required*]						# FLM_REQUIRED_TAG - key value is required
  [ 2 unique]							# FLM_UNIQUE_TAG - key has only 1 reference
  { 2 <field> }...					# FLM_FIELD_TAG - compound key if 2 or more
	 [ 3 case mixed | upper]		# FLM_CASE_TAG - text-only, define chars case
	 [ 3 <field>]...					# FLM_FIELD_TAG - alternate field(s)
	 [ 3 paired]						# FLM_PAIRED_TAG - add field ID to key
	 [ 3 optional*						# FLM_OPTIONAL_TAG - component's value is optional
	 | 3 required]						# FLM_REQUIRED_TAG - component's value is required
	 [ 3 use eachword|value|field|minspaces|nounderscore|nospace|nodash] # FLM_USE_TAG

<field> ==
  n field <field path>				#  path identifies field -- maybe "based"
  [ m type <data type>]				# FLM_TYPE_TAG - only for ixing unregistered fields
	
Please Note:	This code only supports the minimal old 11 index format
					needed for skads databases.			
****************************************************************************/
FSTATIC RCODE DDIxParse(
	TDICT *			pTDict,
	DDENTRY *		pDDEntry,					// Points to defined entry.
	FlmRecord *		pRecord,						// Index definition record.
	void *			pvField)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiIfdFlags;
	FLMUINT			uiTempIfdFlags;
	FLMUINT			uiBaseNum;
	FLMUINT			uiNLen;
	TIXD *			pTIxd;
	TIFD * 			pLastTIfd;
	TIFD *			pTIfd;
	void *			pvTempField = NULL;
	void *			pvIfdField = NULL;
	char				szNativeBuf[ 64];
	FLMUINT			uiCompoundPos;
	FLMUINT			uiTemp;
	FLMBOOL			bHasRequiredTag = TRUE;
	FLMBOOL			bOld11Mode = FALSE;

	if( (pTIxd = (TIXD *) GedPoolAlloc( &pTDict->pool, sizeof( TIXD))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pTIxd->pNextTIfd = NULL;
	pTIxd->uiFlags = 0;
	pTIxd->uiContainerNum = FLM_DATA_CONTAINER;
	pTIxd->uiNumFlds = 0;
	pTIxd->uiLanguage = pTDict->uiDefaultLanguage;
	pTIxd->uiEncId = 0;

	if( (pvField = pRecord->firstChild( pRecord->root())) == NULL)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	pLastTIfd = NULL;
	for( ; pvField; pvField = pRecord->nextSibling( pvField))
	{
		switch ( pRecord->getFieldID( pvField))
		{
			case	FLM_CONTAINER_TAG:
			{
				char 		szTmpBuf [50];
				FLMUINT	uiLen = sizeof( szTmpBuf);
	
				// See if a special keyword is used - ALL or *
	
				if ((pRecord->getDataType( pvField) == FLM_TEXT_TYPE) &&
					 (RC_OK( pRecord->getNative( pvField, szTmpBuf, &uiLen))) &&
					 (f_stricmp( "ALL", szTmpBuf) == 0 ||
					  f_stricmp( "*", szTmpBuf) == 0))
				{
					if (pTDict->pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_50)
					{
						rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
						goto Exit;
					}
	
					// Zero will mean all containers
	
					pTIxd->uiContainerNum = 0;
				}
				else
				{
					if( RC_BAD( rc = DDGetReference(  pRecord, pvField, NULL,
												&pTIxd->uiContainerNum)))
					{
						goto Exit;
					}
					if( pTIxd->uiContainerNum == 0)
					{
						pTIxd->uiContainerNum = FLM_DATA_CONTAINER;
					}
				}
				break;
			}

			case	FLM_COUNT_TAG:
				pTIxd->uiFlags |= IXD_COUNT;
				break;

			case	FLM_LANGUAGE_TAG:
				uiNLen = sizeof( szNativeBuf);
				(void) pRecord->getNative( pvField, szNativeBuf, &uiNLen);
				pTIxd->uiLanguage = FlmLanguage( szNativeBuf);
				break;


			case	FLM_ENCDEF_TAG:
			{
				uiNLen = sizeof( szNativeBuf);
				(void) pRecord->getNative( pvField, szNativeBuf, &uiNLen);
				pTIxd->uiEncId = f_atoud( szNativeBuf);
				flmAssert( pTIxd->uiEncId);
				break;
			}

			case	FLM_TYPE_TAG:
				// Is only compound for NDS definitions.  This parsers default.
				bOld11Mode = TRUE;
				break;

			case	FLM_POSITIONING_TAG:
				if (pTDict->pDb->pFile->FileHdr.uiVersionNum >= FLM_VER_4_3)
				{
					pTIxd->uiFlags |= IXD_POSITIONING;
				}
				else
				{
	
					// Positioning indexes not allowed prior to 4.3
	
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
				break;

			case	FLM_FIELD_TAG:
				uiCompoundPos = 0;
				uiBaseNum = 0;
				uiIfdFlags = IFD_FIELD;
				bHasRequiredTag = TRUE;
				pvTempField = pvField;
				bOld11Mode = TRUE;
				goto Parse_Fields;

			case	FLM_KEY_TAG:
				uiCompoundPos = 0;
				uiBaseNum = 0;
				uiIfdFlags = IFD_FIELD | IFD_OPTIONAL;
				bHasRequiredTag = FALSE;
	
				uiNLen = sizeof( szNativeBuf);
				(void) pRecord->getNative( pvField, szNativeBuf, &uiNLen);
	
				if( f_strnicmp( szNativeBuf, "EACH", 4) == 0)
				{
					pTIxd->uiFlags |= IXD_EACHWORD;
					uiIfdFlags = IFD_EACHWORD | IFD_OPTIONAL;
				}
	
				if( (pvTempField = pRecord->firstChild( pvField)) == NULL)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
Parse_Fields:			
				for( ; pvTempField; pvTempField = pRecord->nextSibling( pvTempField))
				{
					switch( pRecord->getFieldID( pvTempField))
					{
					case	FLM_BASE_TAG:
						if( RC_BAD( rc = DDGetReference( pRecord, 
								pvTempField, NULL, &uiBaseNum)))
						{
							goto Exit;
						}
						break;
	
					case	FLM_COMBINATIONS_TAG:
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
	
					case	FLM_POST_TAG:
						pTIxd->uiFlags |= IXD_HAS_POST;
						uiIfdFlags |= IFD_POST;
						break;
	
					case	FLM_REQUIRED_TAG:			// Default - doesn't mean anything
						break;
	
					case	FLM_OPTIONAL_TAG:
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
	
					case  FLM_UNIQUE_TAG :
						pTIxd->uiFlags |= IXD_UNIQUE;
						uiIfdFlags |= IFD_UNIQUE_PIECE;	// Set the Unique Index Flag
						break;
	
					case	FLM_FIELD_TAG:
						pTIxd->uiNumFlds++;
	
						if( bOld11Mode)
						{
							pvField = pvTempField;
						}
	
						// Need to set IFD_COMPOUND if there is more than one field.
	
						if( pTIxd->uiNumFlds == 1 &&
							(pRecord->find( pvTempField, FLM_FIELD_TAG, 2) != NULL))
						{
							uiIfdFlags |= IFD_COMPOUND;
						}
	
						pTIfd = pLastTIfd;
						if( RC_BAD(rc = DDBuildFldPath( pTDict, &pLastTIfd, 
							pRecord, pvTempField, uiBaseNum)))
						{
							goto Exit;
						}
	
						pLastTIfd->uiCompoundPos = uiCompoundPos++;
	
						if( !pTIfd)						// First time?
						{
							pTIxd->pNextTIfd = pLastTIfd;	// Link first IFD
						}
						else
						{
							pTIfd->pNextTIfd = pLastTIfd;
						}
						uiTempIfdFlags = uiIfdFlags;
						if( bOld11Mode)
						{
							// Default is required for each field.
							uiTempIfdFlags &= ~IFD_OPTIONAL;
							uiTempIfdFlags |= (IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET);
						}
					
						for( pvIfdField = pRecord->firstChild( pvTempField); 
							  pvIfdField; pvIfdField = pRecord->nextSibling( pvIfdField))
						{
							switch ( pRecord->getFieldID( pvIfdField))
							{
							//
							// General IFD options only for this field GROUP
							//
							case FLM_CASE_TAG:
								uiNLen = sizeof( szNativeBuf);
								(void) pRecord->getNative( pvIfdField, szNativeBuf, &uiNLen);
	
								if( f_strnicmp( szNativeBuf, "UPPE", 4) == 0)
								{
									uiTempIfdFlags |= IFD_UPPER;
								}
								break;
	
							case FLM_FIELD_TAG:
								break;
	
							case FLM_OPTIONAL_TAG:
								if( bOld11Mode)
								{
									// Old 11 format - default for each field is required.
									uiTempIfdFlags |= IFD_OPTIONAL;
									uiTempIfdFlags &= ~(IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET);
								}
								// New format default is optional
								break;
	
							case FLM_PAIRED_TAG:
								uiTempIfdFlags |= IFD_FIELDID_PAIR;
								break;
	
							case FLM_POST_TAG:
								// FUTURE: Post piece where other pieces are not
								uiTempIfdFlags |= IFD_POST;
								break;
	
							case FLM_REQUIRED_TAG:
								bHasRequiredTag = TRUE;
								uiTempIfdFlags &= ~IFD_OPTIONAL;
								uiTempIfdFlags |= (IFD_REQUIRED_PIECE | IFD_REQUIRED_IN_SET);
								break;
	
							case FLM_LIMIT_TAG:
								if( RC_BAD( pRecord->getUINT( pvIfdField, &uiTemp)) || 
									uiTemp > IFD_DEFAULT_LIMIT)
								{
									pLastTIfd->uiLimit = IFD_DEFAULT_LIMIT;
								}
								else
								{
									pLastTIfd->uiLimit = uiTemp;
								}
								break;
	
							case FLM_UNIQUE_TAG:
								// FUTURE: option to select specific unique fields.
								uiTempIfdFlags |= IFD_UNIQUE_PIECE;
								pTIxd->uiFlags |= IXD_UNIQUE;
								break;
	
							case FLM_USE_TAG:
								// All these are exclusive values. Take the last value.
								uiNLen = sizeof( szNativeBuf);
								(void) pRecord->getNative( pvIfdField, szNativeBuf, &uiNLen);
	
								if( f_strnicmp( szNativeBuf, "EACH", 4) == 0)
								{
									uiTempIfdFlags |= IFD_EACHWORD;
									uiTempIfdFlags &= ~(IFD_VALUE|IFD_SUBSTRING);
								}
								else if( f_strnicmp( szNativeBuf, "SUBS", 4) == 0)
								{
									pTIxd->uiFlags |= IXD_HAS_SUBSTRING;
									uiTempIfdFlags |= IFD_SUBSTRING;
									uiTempIfdFlags &= ~(IFD_VALUE|IFD_EACHWORD);
									if( pLastTIfd->uiLimit == IFD_DEFAULT_LIMIT)
									{
										pLastTIfd->uiLimit = IFD_DEFAULT_SUBSTRING_LIMIT;
									}
								}
								else if( f_strnicmp( szNativeBuf, "VALU", 4) == 0)
								{
									uiTempIfdFlags |= IFD_VALUE;
									uiTempIfdFlags &= ~(IFD_EACHWORD|IFD_SUBSTRING);
								}
								else if( f_strnicmp( szNativeBuf, "FIEL", 4) == 0)
								{
									uiTempIfdFlags |= IFD_CONTEXT;
									uiTempIfdFlags &= ~(IFD_VALUE|IFD_EACHWORD|IFD_SUBSTRING);
								}
								break;
									
							case FLM_FILTER_TAG:
								uiNLen = sizeof( szNativeBuf);
								(void) pRecord->getNative( pvIfdField, szNativeBuf, &uiNLen);
	
								if( f_strnicmp( szNativeBuf, "MINS", 4) == 0)
								{
									uiTempIfdFlags |= IFD_MIN_SPACES;
								}
								else if( f_strnicmp( szNativeBuf, "NOUN", 4) == 0)
								{
									uiTempIfdFlags |= IFD_NO_UNDERSCORE;
								}
								else if( f_strnicmp( szNativeBuf, "NOSP", 4) == 0)
								{
									uiTempIfdFlags |= IFD_NO_SPACE;
								}
								else if( f_strnicmp( szNativeBuf, "NODA", 4) == 0)
								{
									uiTempIfdFlags |= IFD_NO_DASH;
								}
								else
								{
									rc = RC_SET( FERR_SYNTAX);
									goto Exit;
								}
								break;
	
							default:
								if( pRecord->getFieldID( pvIfdField) < FLM_UNREGISTERED_TAGS &&
									pRecord->getFieldID( pvIfdField) != FLM_COMMENT_TAG)
								{
									rc = RC_SET( FERR_SYNTAX);
									goto Exit;
								}
								break;
							} // end switch
						} // end for loop parsing all level 3 tags
	
						// Parse again the level 3 field definitions.  Now we
						// have the IFD uiFlags value to assign each piece that
						// will have the same compound position.
	
						pLastTIfd->uiFlags |= uiTempIfdFlags;
	
						for( pvIfdField = pRecord->firstChild( pvTempField); 
							  pvIfdField; pvIfdField = pRecord->nextSibling( pvIfdField))
						{
							if( pRecord->getFieldID( pvIfdField) == FLM_FIELD_TAG )
							{
								rc = RC_SET( FERR_SYNTAX);
								goto Exit;
							}
						}
						break;	// Done parsing "2 field xx yy zz"
	
					default:
						if( bOld11Mode)
						{
							break;
						}
	
						if( pRecord->getFieldID( pvTempField) < FLM_UNREGISTERED_TAGS &&
							pRecord->getFieldID( pvTempField) != FLM_COMMENT_TAG)
	
						{
							rc = RC_SET( FERR_SYNTAX);
							goto Exit;
						}
						break;
					} // end switch
	
				} // end for loop

				// Special case for optional 
				if( !bHasRequiredTag)
				{
					// Set all of the IFD flags to IFD_REQUIRED_IN_SET
					for( pTIfd = pTIxd->pNextTIfd; pTIfd; pTIfd = pTIfd->pNextTIfd)
					{
						pTIfd->uiFlags |= IFD_REQUIRED_IN_SET;
					}
				}
				break;

			default:
				if( pRecord->getFieldID( pvField) < FLM_UNREGISTERED_TAGS &&
					pRecord->getFieldID( pvField) != FLM_COMMENT_TAG)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
				break;
		}
	}
	pDDEntry->vpDef = (void *) pTIxd;

Exit:

	if( RC_BAD(rc))
	{
		if( pvIfdField)
			pTDict->uiBadField = pRecord->getFieldID( pvIfdField);
		else if( pvTempField)
			pTDict->uiBadField = pRecord->getFieldID( pvTempField);
		else if( pvField)
			pTDict->uiBadField = pRecord->getFieldID( pvField);
	}
	else
	{
		pTDict->uiNewIxds++;
		pTDict->uiNewIfds += pTIxd->uiNumFlds;
		pTDict->uiNewLFiles++;
	}
	return( rc );
}

/****************************************************************************
Desc:		Build field path for each index field. This function will also
			check for the existence of the 'batch' option for QF indexes.
****************************************************************************/
FSTATIC RCODE DDBuildFldPath(
	TDICT *			pTDict,
	TIFD **			ppTIfd,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiBaseNum)
{
	RCODE				rc = FERR_OK;
	TIFD *			pTIfd;
	TIFP *			pLastFldPath;
	TIFP *			pTIfp;
	FLMUINT			uiNumInFldPath;
	char				szNameBuf[ 32 ];
	char *			pszCurrent;
	char				szNativeBuf[ FDD_MAX_VALUE_SIZE];
	FLMUINT			uiBufLen;
	FLMUINT			uiPos;

	pTDict->uiTotalIfds++;
	if( (pTIfd = (TIFD *) GedPoolAlloc( &pTDict->pool,	sizeof( TIFD))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pTIfd->pTIfp = NULL;
	pTIfd->pNextTIfd = NULL;
	pTIfd->uiFlags = 0;
	pTIfd->uiNextFixupPos = 0;
	pTIfd->uiLimit = IFD_DEFAULT_LIMIT;
	pTIfd->uiCompoundPos = 0;

	pLastFldPath = NULL;
	*ppTIfd = pTIfd;

	// Build the field paths

	DDTextToNative( pRecord, pvField, szNativeBuf,
		FDD_MAX_VALUE_SIZE, &uiBufLen);

	pszCurrent = szNativeBuf;
	uiNumInFldPath = uiPos = 0;

	if( uiBaseNum )
	{
		uiNumInFldPath++;
		if( (pTIfp = (TIFP *) GedPoolAlloc( &pTDict->pool,
			sizeof( TIFP ))) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		pTIfp->pNextTIfp = NULL;
		pTIfp->bFieldInThisDict = FALSE;
		pTIfp->uiFldNum = uiBaseNum;
		pTIfd->pTIfp = pTIfp;
		pLastFldPath = pTIfp;
	}

	while( uiPos < uiBufLen)
	{
		uiNumInFldPath++;
		if( DDMoveWord( szNameBuf, pszCurrent, 
				sizeof( szNameBuf ), &uiPos ) == FALSE )
		{
			break;
		}

		if( (pTIfp = (TIFP *) GedPoolAlloc( &pTDict->pool,
			sizeof( TIFP ))) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		pTIfp->pNextTIfp = NULL;
		pTIfp->bFieldInThisDict = FALSE;

		if( pTIfd->pTIfp == NULL)
		{
			pTIfd->pTIfp = pTIfp;
		}
		else
		{
			pLastFldPath->pNextTIfp = pTIfp;
		}

		pLastFldPath = pTIfp;

		// See if there is a wildcard in the path.

		if (f_stricmp( szNameBuf, "*") == 0)
		{
			if (pTDict->pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_50)
			{
				rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
				goto Exit;
			}
			else
			{
				pTIfp->uiFldNum = FLM_ANY_FIELD;
			}
		}
		else
		{
			if( RC_BAD( rc = DDGetReference( NULL, NULL, szNameBuf,
									&pTIfp->uiFldNum)))
			{
				goto Exit;
			}
		}
	}

	if( uiNumInFldPath == 0)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	// Cannot have wildcard in last field of field path.

	if (pLastFldPath->uiFldNum == FLM_ANY_FIELD)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	// Single field has the field NULL terminated
	
	if( uiNumInFldPath == 1 )
	{
		pTDict->uiNewFldPaths += 2;
	}
	else
	{
		// The field paths are stored child to parent and parent to child
		// each are zero terminated.

		pTDict->uiNewFldPaths += 2 * (uiNumInFldPath + 1);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Parse a data dictionary domain definition for correct syntax &
		assign the correct attributes.
****************************************************************************/
FSTATIC RCODE	DDContainerParse(
	TDICT *		pTDict,
	DDENTRY *	pDDEntry,
	FlmRecord *	pRecord)
{
	RCODE    	rc = FERR_OK;
	void *		pvField = NULL;

	if( pDDEntry)
	{

		if( (pvField = pRecord->firstChild( pRecord->root())) != NULL)
		{
			for( ; pvField; pvField = pRecord->nextSibling( pvField))
			{
				// Only option is unregistered fields

				if( pRecord->getFieldID( pvField) < FLM_FREE_TAG_NUMS)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
		}
	}

Exit:	

	if( RC_BAD(rc) && pvField)
	{
		pTDict->uiBadField = pRecord->getFieldID( pvField);
	}

	return( rc );
}

/****************************************************************************
Desc:	Parse a data dictionary domain definition for correct syntax &
		assign the correct attributes.
****************************************************************************/
FSTATIC RCODE	DDEncDefParse(
	TDICT *		pTDict,
	DDENTRY *	pDDEntry,
	FlmRecord *	pRecord,
	FLMUINT		uiDictRecNum)
{
	RCODE    	rc = FERR_OK;
	void *		pvField = NULL;
	TENCDEF *	pTEncDef;

	// Make sure the version of the database is correct for encryption.
	
	if (pTDict->pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_60)
	{
		rc = RC_SET( FERR_UNSUPPORTED_FEATURE);
		goto Exit;
	}

	if( (pTEncDef = (TENCDEF *)GedPoolAlloc( &pTDict->pool,
		sizeof(TENCDEF))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pTEncDef->uiRecNum = uiDictRecNum;
	pTEncDef->uiAlgType = 0;
	pTEncDef->uiState = 0;
	pTEncDef->pucKeyInfo = NULL;
	pTEncDef->uiLength = 0;

	if( pDDEntry)
	{

		if( (pvField = pRecord->firstChild( pRecord->root())) != NULL)
		{
			for( ; pvField; pvField = pRecord->nextSibling( pvField))
			{
				switch ( pRecord->getFieldID( pvField) )
				{
					case FLM_TYPE_TAG:
					{
						// Get the encryption type.
						if (RC_BAD( rc = DDGetEncType( pRecord,
												 pvField,
												 &pTEncDef->uiAlgType)))
						{
							goto Exit;
						}
						break;
					}

					case FLM_KEY_TAG:
					{
						// Get the key information.
						if (RC_BAD( rc = DDGetEncKey( pTDict,
											   pRecord,
											   pvField,
											   pTEncDef)))
						{
							goto Exit;
						}
						break;
					}

					case FLM_STATE_TAG:
					{
						// Get the status information.
						if (RC_BAD( rc = DDParseStateOptions( pRecord,
														  pvField,
														  &pTEncDef->uiState)))
						{
							goto Exit;
						}
						break;
					}

					default:
					{
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
					}
				}
			}

			pDDEntry->vpDef = (void *)pTEncDef;

		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:

	if (RC_BAD( rc) && pvField)
	{
		pTDict->uiBadField = pRecord->getFieldID( pvField);
	}

	return( rc );
}

/****************************************************************************
Desc:		Move word delimited by spaces from src to dest.  Used to move a
			word at a time for field path lists.
Notes:	Isolated so changes can be made to delemeting NAMES.
Visit:	Still bugs when name > buffer size - won't happen because only #'s
****************************************************************************/
FSTATIC FLMBOOL DDMoveWord(
	char *		pucDest,
	char *		pucSrc,
	FLMUINT		uiMaxDestLen,
	FLMUINT *	puiPos)
{
	FLMBOOL		bFoundWord = TRUE;
	FLMUINT		uiPos = *puiPos;
	char *		pMatch;
	FLMUINT		uiBytesToCopy;

	pucSrc += uiPos;
	while( *pucSrc == NATIVE_SPACE)
	{
		pucSrc++;
	}

	pMatch = pucSrc;
	while( *pMatch > NATIVE_SPACE)
	{
		pMatch++;
	}

	if( !*pMatch)
	{
		if( *pucSrc == '\0')
		{
			bFoundWord = FALSE;
			goto Exit;
		}
		
		uiBytesToCopy = f_strlen( pucSrc);
		
		if( uiBytesToCopy + 1 > uiMaxDestLen)
		{
			uiBytesToCopy = uiMaxDestLen - 1;
		}
		
		f_memcpy( pucDest, pucSrc, uiBytesToCopy + 1);
		*puiPos = uiPos + uiBytesToCopy + 1;
	}
	else
	{
		// Copy the bytes between pucSrc and pMatch minus one
		
		uiBytesToCopy = (FLMUINT) (pMatch - pucSrc);
		
		if( uiBytesToCopy + 1 > uiMaxDestLen)
		{
			uiBytesToCopy = uiMaxDestLen - 1;
		}

		f_memcpy( pucDest, pucSrc, uiBytesToCopy );
		pucDest[ uiBytesToCopy ] = '\0';

		// Go past consuctive spaces

		while( pucSrc[ ++uiBytesToCopy ] == NATIVE_SPACE)
		{
			uiBytesToCopy++;
		}

		*puiPos = uiPos + uiBytesToCopy;
	}
	
Exit:

	return( bFoundWord);
}

/****************************************************************************
Desc: 	Normalizes an internal string with possible formatting codes into
			a NATIVE string.  Drops all formatting codes and extended chars.
****************************************************************************/
FSTATIC void DDTextToNative(
	FlmRecord *		pRecord,
	void *			pvField,
	char *			pszBuffer,
	FLMUINT			uiBufLen,
	FLMUINT *		puiBufLen)
{
	RCODE			rc = FERR_OK;

	pszBuffer[ 0] = 0;
	
	if( pRecord->getDataLength( pvField))
	{
		if( RC_BAD( rc = pRecord->getNative( pvField, pszBuffer, &uiBufLen)))
		{
			if( rc != FERR_CONV_DEST_OVERFLOW)
			{
				pszBuffer[0] = 0;
				uiBufLen = 0;
			}
		}
	}
	else
	{
		uiBufLen = 0;
	}

	if( puiBufLen)
	{
		// Length needs to include the null byte
		
		*puiBufLen = uiBufLen + 1;
	}

	return;
}
