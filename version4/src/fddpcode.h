//-------------------------------------------------------------------------
// Desc:	Typedefs for strucures needed to build pcode.
// Tabs:	3
//
//		Copyright (c) 1991-1992,1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fddpcode.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FDDPCODE_H
#define FDDPCODE_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Logical File Save Area Layout for 4.x files.

#define LFH_LF_NUMBER_OFFSET	0		// Logical file number
#define LFH_TYPE_OFFSET			2	 	// Type of logical file
#define LFH_STATUS_OFFSET		3		// Contains status bits
#define LFH_ROOT_BLK_OFFSET	4		// B-TREE root block address
//#define LFH_FUTURE1			8		// Not necessarily zeroes - Code bases
												// 31 and 40 put stuff here.
#define LFH_NEXT_DRN_OFFSET	12		// Next DRN for containers
#define LFH_MAX_FILL_OFFSET	16		// Max fill % after rightmost split.
#define LFH_MIN_FILL_OFFSET	17		// Min fill % in blk after normal delete
//#define LFH_FUTURE2			18		// Filled with zeros
#define LFH_SIZE					32		// Maximum size of LFH.


typedef struct DDEntry          	 *	DDENTRY_p;
typedef struct TmpFieldDef			 *	TFIELD_p;
typedef struct TmpIndexFieldPath  *	TIFP_p;
typedef struct TmpIndexFieldDef	 *	TIFD_p;
typedef struct TmpIndexDef			 *	TIXD_p;
typedef struct Tmp_Dictionary		 *	TDICT_p;
typedef struct TmpFlaimArea		 *	TFAREA_p;

RCODE fdictRebuild(
	FDB *					pDb);

RCODE	fdictBuildTables(
	TDICT_p				pTDict,
	FLMBOOL				bRereadLFiles,
	FLMBOOL				bNewDict);

RCODE fdictInitTDict(
	FDB *					pDb,
	TDICT_p				pTDict);

RCODE fdictCopySkeletonDict(
	FDB *					pDb);

RCODE fdictCloneDict(
	FDB *					pDb);

RCODE fdictFixupLFileTbl(
	FDICT *				pDict);

RCODE fdictProcessAllDictRecs( 
	FDB *					pDb,
	TDICT_p				pTDict);

RCODE fdictProcessRec( 
	TDICT_p				pTDict,
	FlmRecord *			pRecord,
	FLMUINT				uiDictRecNum);

RCODE DDGetFieldType(  
	FlmRecord *			pRecord,
	void *				pvField,
	FLMUINT *			puiFldInfo);

RCODE DDGetEncType(
	FlmRecord *			pRecord,
	void *				pvField,
	FLMUINT *			puiFldInfo);

RCODE fdictCreateNewDict(
	FDB *					pDb);

RCODE  fdictCreate( 
	FDB *					pDb,
	const char *		pszDictPath,
	const char *		pDictBuf);

RCODE flmAddRecordToDict( 
	FDB *					pDb,
	FlmRecord *			pRecord,
	FLMUINT				uiDictId,
	FLMBOOL				bRereadLFiles);

/****************************************************************************
Desc:	Structure for type, DRN and name for data dictionary entries
****************************************************************************/
typedef struct DDEntry
{
	DDENTRY_p 	pNextEntry;						
	void *		vpDef;	
	FLMUINT  	uiEntryNum;			
	FLMUINT   	uiType;				
} DDENTRY;

/****************************************************************************
Desc:	Temporary field info used during a database create or dictionary
		modification.  This field is pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TmpFieldDef
{
	FLMUINT		uiFldNum;
	FLMUINT		uiFldInfo;
} TFIELD;

/****************************************************************************
Desc:	Temporary encryption definition info used during a database create or
		dictionary modification.  This field is pointed to by the
		DDEntry structure.
****************************************************************************/
typedef struct
{
	FLMUINT		uiRecNum;
	FLMUINT		uiState;
	FLMUINT		uiAlgType;
	FLMBYTE *	pucKeyInfo;
	FLMUINT		uiLength;
} TENCDEF;

/****************************************************************************
Desc:		Used as temporary storage for index definitions during a
			database create or dictionary modification.  This field is
			pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TmpIndexFieldPath
{
	TIFP_p		pNextTIfp;			// Linked list of IFPs							
	FLMBOOL		bFieldInThisDict;	// Was field reference found in the
											//	dictionary we are updating? 
	FLMUINT		uiFldNum;			// Fixedup field ID value						
} TIFP;

/****************************************************************************
Desc:		Used as temporary storage for index definitions during a
			database create or dictionary modification.  This field is
			pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TmpIndexFieldDef
{
	TIFP_p 		pTIfp;		  		// Linked list of temporary field paths	
	TIFD_p		pNextTIfd;			// Linked List										
	FLMUINT		uiFlags;				// Field type & processing flags				
	FLMUINT		uiNextFixupPos;	// Next fixup position  
	FLMUINT		uiLimit;				// Zero or limit of characters/bytes		
	FLMUINT		uiCompoundPos;		// Position of this field is in 
											//	the compound key.  Zero based number.	
} TIFD;

/****************************************************************************
Desc:		Used as temporary storage for index definitions during a
			database create or dictionary modification.  This field is
			pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TmpIndexDef
{
	TIFD_p   	pNextTIfd;  		// Linked list of TIFDs							
	FLMUINT		uiFlags;				// Index attributes								
	FLMUINT		uiContainerNum;	// Container number of data records			
	FLMUINT		uiNumFlds;			// Number of field definitions				
	FLMUINT		uiLanguage;			// Index language
	FLMUINT		uiEncId;				// Encryption Definition
} TIXD;

/****************************************************************************
Desc:		Contains the dictionary entries through parsing all of the dictionary 
			records. Used for expanding record definitions, checking index 
			definitions, building fixup position values and last of all 
			BUILDING THE PCODE.
****************************************************************************/
typedef struct Tmp_Dictionary
{
	FDB *			pDb;
	POOL       	pool;					// Pool for the DDENTRY allocations.
	LFILE *		pLFile;				// Dictionary container LFile
	FDICT_p		pDict;				// Pointer to new dictionary.
	FLMBOOL		bWriteToDisk;		// Flag indicating if PCODE should be
											//	written to disk after being generated.

	// Variables for building dictionaries 

	FLMUINT		uiCurPcodeAddr;	// Current pcode block we are adding to
	FLMUINT		uiBlockSize;		// PCODE Block size

	// Used in building the temporary structures 
	
	FLMUINT		uiVersionNum;		// Version number of database.
	DDENTRY_p	pFirstEntry;
	DDENTRY_p	pLastEntry;

	FLMUINT		uiNewIxds;
	FLMUINT		uiNewIfds;
	FLMUINT		uiNewFldPaths;
	FLMUINT		uiNewLFiles;

	FLMUINT		uiTotalItts;
	FLMUINT		uiTotalIxds;
	FLMUINT		uiTotalIfds;
	FLMUINT		uiTotalFldPaths;
	FLMUINT		uiTotalLFiles;

	FLMUINT		uiBadField;			// Set to field number on most errors.
	FLMUINT		uiBadReference;	// Same

	FLMUINT		uiDefaultLanguage;// Default language to set in each index.
} TDICT;

#include "fpackoff.h"

#endif
