//-------------------------------------------------------------------------
// Desc:	I/O for Netware OS
// Tabs:	3
//
//		Copyright (c) 1998-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fnlm.cpp 12362 2006-03-09 12:11:37 -0700 (Thu, 09 Mar 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_NLM)

#define F_IO_MAX_CREATION_TRIES   			10
#define zMAX_COMPONENT_NAME					256
#define zGET_INFO_VARIABLE_DATA_SIZE		(zMAX_COMPONENT_NAME * 2)

#define zOK												0			// the operation succeeded
#define zERR_NO_MEMORY								20000 	// insufficent memory to complete the request
#define zERR_NOT_SUPPORTED							20011 	// the operation is not supported
#define zERR_CONNECTION_NOT_LOGGED_IN			20007 	// the connection has not been logged in
#define zERR_END_OF_FILE							20100 	// read past the end of file
#define zERR_OUT_OF_SPACE							20103 	// no available disk space is left
#define zERR_BAD_FILE_HANDLE						20401 	// the file handle is out of range, bad instance, or doesn't exist
#define zERR_INVALID_NAME							20403		// path name is invalid -- bad syntax
#define zERR_INVALID_CHAR_IN_NAME				20404 	// path name had an invalid character
#define zERR_INVALID_PATH							20405 	// the path is syntactically incorrect
#define zERR_NAME_NOT_FOUND_IN_DIRECTORY		20407 	// name does not exist in the direcory being searched
#define zERR_NO_NAMES_IN_PATH						20409 	// a NULL file name was given
#define zERR_NO_MORE_NAMES_IN_PATH 				20410 	// doing a wild search but ran out of names to search
#define zERR_PATH_MUST_BE_FULLY_QUALIFIED		20411 	// path name must be fully qualified in this context
#define zERR_FILE_ALREADY_EXISTS					20412 	// the given file already exists
#define zERR_NAME_NO_LONGER_VALID				20413 	// the dir/file name is no longer valid
#define zERR_DIRECTORY_NOT_EMPTY					20417 	// the directory still has files in it
#define zERR_NO_FILES_FOUND						20424 	// no files matched the given wildcard pattern
#define zERR_DIR_CANNOT_BE_OPENED				20435 	// the requested parent was not found
#define zERR_NO_OPEN_PRIVILEGE					20438 	// no the right privileges to open the file
#define zERR_NO_MORE_CONTEXT_HANDLE_IDS		20439 	// there are no more available context handle IDs
#define zERR_INVALID_PATH_FORMAT					20441 	// the pathFormat is either invalid or unsupported
#define zERR_ALL_FILES_IN_USE						20500 	// all files were in use
#define zERR_SOME_FILES_IN_USE					20501 	// some of the files were in use
#define zERR_ALL_FILES_READ_ONLY					20502 	// all files were READONLY
#define zERR_SOME_FILES_READ_ONLY				20503 	// some of the files were READONLY
#define zERR_ALL_NAMES_EXIST						20504 	// all of the names already existed
#define zERR_SOME_NAMES_EXIST						20505 	// some of the names already existed
#define zERR_NO_RENAME_PRIVILEGE					20506 	// you do not have privilege to rename the file
#define zERR_RENAME_DIR_INVALID					20507 	// the selected directory may not be renamed
#define zERR_RENAME_TO_OTHER_VOLUME				20508 	// a rename/move may not move the beast to a different volume
#define zERR_CANT_RENAME_DATA_STREAMS			20509 	// not allowed to rename a data stream
#define zERR_FILE_RENAME_IN_PROGRESS			20510 	// the file is already being renamed by a different process
#define zERR_CANT_RENAME_TO_DELETED				20511 	// only deleted files may be renamed to a deleted state
#define zERR_HOLE_IN_DIO_FILE  	            20651 	// DIO files cannot have holes
#define zERR_BEYOND_EOF  	            		20652 	// DIO files cannot be read beyond EOF
#define zERR_INVALID_PATH_SEPARATOR				20704 	// the name space does not support the requested path separator type
#define zERR_VOLUME_SEPARATOR_NOT_SUPPORTED	20705 	// the name space does not support volume separators
#define zERR_BAD_VOLUME_NAME   					20800 	// the given volume name is syntactically incorrect
#define zERR_VOLUME_NOT_FOUND  					20801 	// the given volume name could not be found
#define zERR_NO_SET_PRIVILEGE  					20850 	// does not have rights to modify metadata
#define zERR_NO_CREATE_PRIVILEGE					20851		// does not have rights to create an object
#define zERR_ACCESS_DENIED							20859 	// authorization/attributes denied access
#define zERR_NO_WRITE_PRIVILEGE					20860 	// no granted write privileges
#define zERR_NO_READ_PRIVILEGE					20861 	// no granted read privileges
#define zERR_NO_DELETE_PRIVILEGE					20862 	// no delete privileges
#define zERR_SOME_NO_DELETE_PRIVILEGE			20863 	// on wildcard some do not have delete privileges
#define zERR_NO_SUCH_OBJECT						20867 	// no such object in the naming services
#define zERR_CANT_DELETE_OPEN_FILE				20868 	// cant delete an open file without rights
#define zERR_NO_CREATE_DELETE_PRIVILEGE		20869 	// no delete on create privileges
#define zERR_NO_SALVAGE_PRIVILEGE				20870 	// no privileges to salvage this file
#define zERR_FILE_READ_LOCKED						20905 	// cant grant read access to the file
#define zERR_FILE_WRITE_LOCKED					20906 	// cant grant write access to the file

#define zRR_READ_ACCESS								0x00000001
#define zRR_WRITE_ACCESS							0x00000002
#define zRR_DENY_READ								0x00000004
#define zRR_DENY_WRITE								0x00000008
#define zRR_SCAN_ACCESS								0x00000010
#define zRR_ENABLE_IO_ON_COMPRESSED_DATA		0x00000100
#define zRR_LEAVE_FILE_COMPRESSED	        	0x00000200
#define zRR_DELETE_FILE_ON_CLOSE					0x00000400
#define zRR_FLUSH_ON_CLOSE							0x00000800
#define zRR_PURGE_IMMEDIATE_ON_CLOSE			0x00001000
#define zRR_DIO_MODE									0x00002000
#define zRR_ALLOW_SECURE_DIRECTORY_ACCESS		0x00020000
#define zRR_TRANSACTION_ACTIVE					0x00100000
#define zRR_READ_ACCESS_TO_SNAPSHOT				0x04000000
#define zRR_DENY_RW_OPENER_CAN_REOPEN			0x08000000
#define zRR_CREATE_WITHOUT_READ_ACCESS			0x10000000
#define zRR_OPENER_CAN_DELETE_WHILE_OPEN		0x20000000
#define zRR_CANT_DELETE_WHILE_OPEN				0x40000000
#define zRR_DONT_UPDATE_ACCESS_TIME				0x80000000

#define zFA_READ_ONLY		 						0x00000001
#define zFA_HIDDEN 									0x00000002
#define zFA_SYSTEM 									0x00000004
#define zFA_EXECUTE									0x00000008
#define zFA_SUBDIRECTORY	 						0x00000010
#define zFA_ARCHIVE									0x00000020
#define zFA_SHAREABLE		 						0x00000080
#define zFA_SMODE_BITS		 						0x00000700
#define zFA_NO_SUBALLOC								0x00000800
#define zFA_TRANSACTION								0x00001000
#define zFA_NOT_VIRTUAL_FILE						0x00002000
#define zFA_IMMEDIATE_PURGE						0x00010000
#define zFA_RENAME_INHIBIT	 						0x00020000
#define zFA_DELETE_INHIBIT	 						0x00040000
#define zFA_COPY_INHIBIT	 						0x00080000
#define zFA_IS_ADMIN_LINK							0x00100000
#define zFA_IS_LINK									0x00200000
#define zFA_REMOTE_DATA_INHIBIT					0x00800000
#define zFA_COMPRESS_FILE_IMMEDIATELY 			0x02000000
#define zFA_DATA_STREAM_IS_COMPRESSED 			0x04000000
#define zFA_DO_NOT_COMPRESS_FILE	  				0x08000000
#define zFA_CANT_COMPRESS_DATA_STREAM 			0x20000000
#define zFA_ATTR_ARCHIVE	 						0x40000000
#define zFA_VOLATILE									0x80000000

#define zNSPACE_DOS									0
#define zNSPACE_MAC									1
#define zNSPACE_UNIX									2
#define zNSPACE_LONG									4
#define zNSPACE_DATA_STREAM						6
#define zNSPACE_EXTENDED_ATTRIBUTE				7
#define zNSPACE_INVALID								(-1)
#define zNSPACE_DOS_MASK							(1 << zNSPACE_DOS)
#define zNSPACE_MAC_MASK							(1 << zNSPACE_MAC)
#define zNSPACE_UNIX_MASK							(1 << zNSPACE_UNIX)
#define zNSPACE_LONG_MASK							(1 << zNSPACE_LONG)
#define zNSPACE_DATA_STREAM_MASK					(1 << zNSPACE_DATA_STREAM)
#define zNSPACE_EXTENDED_ATTRIBUTE_MASK 		(1 << zNSPACE_EXTENDED_ATTRIBUTE)
#define zNSPACE_ALL_MASK							(0xffffffff)

#define zMODE_VOLUME_ID								0x80000000
#define zMODE_UTF8									0x40000000
#define zMODE_DELETED								0x20000000
#define zMODE_LINK									0x10000000

#define zCREATE_OPEN_IF_THERE						0x00000001
#define zCREATE_TRUNCATE_IF_THERE				0x00000002
#define zCREATE_DELETE_IF_THERE					0x00000004

#define zMATCH_ALL_DERIVED_TYPES					0x00000001
#define zMATCH_HIDDEN								0x1
#define zMATCH_NON_HIDDEN							0x2
#define zMATCH_DIRECTORY							0x4
#define zMATCH_NON_DIRECTORY						0x8
#define zMATCH_SYSTEM								0x10
#define zMATCH_NON_SYSTEM							0x20
#define zMATCH_ALL									(~0)

#define zSETSIZE_NON_SPARSE_FILE					0x00000001
#define zSETSIZE_NO_ZERO_FILL						0x00000002
#define zSETSIZE_UNDO_ON_ERR	 					0x00000004
#define zSETSIZE_PHYSICAL_ONLY	 				0x00000008
#define zSETSIZE_LOGICAL_ONLY	 					0x00000010
#define zSETSIZE_COMPRESSED      				0x00000020

#define zMOD_FILE_ATTRIBUTES						0x00000001
#define zMOD_CREATED_TIME							0x00000002
#define zMOD_ARCHIVED_TIME							0x00000004
#define zMOD_MODIFIED_TIME							0x00000008
#define zMOD_ACCESSED_TIME							0x00000010
#define zMOD_METADATA_MODIFIED_TIME				0x00000020
#define zMOD_OWNER_ID								0x00000040
#define zMOD_ARCHIVER_ID							0x00000080
#define zMOD_MODIFIER_ID							0x00000100
#define zMOD_METADATA_MODIFIER_ID				0x00000200
#define zMOD_PRIMARY_NAMESPACE					0x00000400
#define zMOD_DELETED_INFO							0x00000800
#define zMOD_MAC_METADATA							0x00001000
#define zMOD_UNIX_METADATA							0x00002000
#define zMOD_EXTATTR_FLAGS							0x00004000
#define zMOD_VOL_ATTRIBUTES						0x00008000
#define zMOD_VOL_NDS_OBJECT_ID					0x00010000
#define zMOD_VOL_MIN_KEEP_SECONDS				0x00020000
#define zMOD_VOL_MAX_KEEP_SECONDS				0x00040000
#define zMOD_VOL_LOW_WATER_MARK					0x00080000
#define zMOD_VOL_HIGH_WATER_MARK					0x00100000
#define zMOD_POOL_ATTRIBUTES						0x00200000
#define zMOD_POOL_NDS_OBJECT_ID					0x00400000
#define zMOD_VOL_DATA_SHREDDING_COUNT			0x00800000
#define zMOD_VOL_QUOTA								0x01000000

/***************************************************************************
Desc:
***************************************************************************/
enum zGetInfoMask_t
{
	zGET_STD_INFO										= 0x1,
	zGET_NAME											= 0x2,
	zGET_ALL_NAMES										= 0x4,
	zGET_PRIMARY_NAMESPACE							= 0x8,
	zGET_TIMES_IN_SECS								= 0x10,
	zGET_TIMES_IN_MICROS								= 0x20,
	zGET_IDS												= 0x40,
	zGET_STORAGE_USED									= 0x80,
	zGET_BLOCK_SIZE									= 0x100,
	zGET_COUNTS											= 0x200,
	zGET_EXTENDED_ATTRIBUTE_INFO					= 0x400,
	zGET_DATA_STREAM_INFO							= 0x800,
	zGET_DELETED_INFO									= 0x1000,
	zGET_MAC_METADATA									= 0x2000,
	zGET_UNIX_METADATA								= 0x4000,
	zGET_EXTATTR_FLAGS								= 0x8000,
	zGET_VOLUME_INFO									= 0x10000,
	zGET_VOL_SALVAGE_INFO							= 0x20000,
	zGET_POOL_INFO										= 0x40000
};

/***************************************************************************
Desc:
***************************************************************************/
enum
{
	zINFO_VERSION_A = 1
};

/***************************************************************************
Desc:
***************************************************************************/
typedef enum FileType_t
{
	zFILE_UNKNOWN,
	zFILE_REGULAR,
	zFILE_EXTENDED_ATTRIBUTE,
	zFILE_NAMED_DATA_STREAM,
	zFILE_PIPE,
	zFILE_VOLUME,
	zFILE_POOL,
	zFILE_MAX
} FileType_t;

/***************************************************************************
Desc:
***************************************************************************/
typedef struct	GUID_t
{
	LONG					timeLow;
	WORD					timeMid;
	WORD					timeHighAndVersion;
	BYTE					clockSeqHighAndReserved;
	BYTE					clockSeqLow;
	BYTE					node[ 6];
} GUID_t;

/***************************************************************************
Desc:
***************************************************************************/
typedef struct zMacInfo_s
{
	BYTE 					finderInfo[32];
	BYTE 					proDOSInfo[6];
	BYTE					filler[2];
	LONG					dirRightsMask;
} zMacInfo_s;

/***************************************************************************
Desc:
***************************************************************************/
typedef struct zUnixInfo_s
{
 	LONG					fMode;
 	LONG					rDev;
 	LONG					myFlags;
 	LONG					nfsUID;
 	LONG 					nfsGID;
	LONG					nwUID;
	LONG					nwGID;
	LONG					nwEveryone;
	LONG					nwUIDRights;
	LONG					nwGIDRights;
	LONG					nwEveryoneRights;
 	BYTE					acsFlags;
 	BYTE					firstCreated;
 	FLMINT16				variableSize;
} zUnixInfo_s;

typedef struct zVolumeInfo_s
{
	GUID_t				volumeID;
   GUID_t				ndsObjectID;
	LONG					volumeState;
	LONG					nameSpaceMask;
	
	struct
	{
		FLMUINT64		enabled;
		FLMUINT64		enableModMask;
		FLMUINT64		supported;
	} features;
	
	FLMUINT64			maximumFileSize;	
	FLMUINT64			totalSpaceQuota;
	FLMUINT64			numUsedBytes;
	FLMUINT64			numObjects;
	FLMUINT64			numFiles;
	LONG					authModelID;
	LONG					dataShreddingCount;
	
	struct
	{
		FLMUINT64		purgeableBytes;
		FLMUINT64		nonPurgeableBytes;
		FLMUINT64		numDeletedFiles;
		FLMUINT64		oldestDeletedTime;
		LONG				minKeepSeconds;
		LONG				maxKeepSeconds;
		LONG				lowWaterMark;
		LONG				highWaterMark;
	} salvage;
	
	struct
	{
		FLMUINT64		numCompressedFiles;
		FLMUINT64		numCompDelFiles;
		FLMUINT64		numUncompressibleFiles;
		FLMUINT64		numPreCompressedBytes;
		FLMUINT64		numCompressedBytes;
	} comp;
	 
} zVolumeInfo_s;

/***************************************************************************
Desc:
***************************************************************************/
typedef struct zPoolInfo_s
{
	GUID_t				poolID;
	GUID_t				ndsObjectID;
	LONG					poolState;
	LONG					nameSpaceMask;
	
	struct 
	{
		FLMUINT64		enabled;
		FLMUINT64		enableModMask;
		FLMUINT64		supported;
	} features;
	
	FLMUINT64			totalSpace;
	FLMUINT64			numUsedBytes;
	FLMUINT64			purgeableBytes;
	FLMUINT64			nonPurgeableBytes;
} zPoolInfo_s;

/***************************************************************************
Desc:
***************************************************************************/
typedef struct zInfo_s
{
	LONG					infoVersion;
	FLMINT				totalBytes;
	FLMINT				nextByte;
	LONG					padding;
	FLMUINT64			retMask;
	
	struct 
	{
		FLMUINT64		zid;
		FLMUINT64		dataStreamZid;
		FLMUINT64		parentZid;
		FLMUINT64		logicalEOF;
		GUID_t			volumeID;
		LONG				fileType;
		LONG				fileAttributes;
		LONG				fileAttributesModMask;
		LONG				padding;
	} std;

	struct
	{
		FLMUINT64		physicalEOF;
    	FLMUINT64		dataBytes;
    	FLMUINT64		metaDataBytes;
	} storageUsed;

	LONG					primaryNameSpaceID;
	LONG 					nameStart;

	struct 
	{
		LONG 				numEntries;
		LONG 				fileNameArray;
	} names;

	struct
	{
		FLMUINT64		created;
		FLMUINT64		archived;
		FLMUINT64		modified;
		FLMUINT64		accessed;
		FLMUINT64		metaDataModified;
	} time;

	struct 
	{
	 	GUID_t 			owner;
    	GUID_t 			archiver;
    	GUID_t 			modifier;
    	GUID_t 			metaDataModifier;
	} id;

	struct 
	{
		LONG	 			size;
		LONG	 			sizeShift;
	} blockSize;

	struct 
	{
		LONG	 			open;
		LONG	 			hardLink;
	} count;

	struct 
	{
		LONG	 			count;
		LONG	 			totalNameSize;
		FLMUINT64		totalDataSize;
	} dataStream;

	struct 
	{
		LONG	 			count;
		LONG	 			totalNameSize;
		FLMUINT64		totalDataSize;
	} extAttr;

	struct 
	{
		FLMUINT64		time;
		GUID_t 			id;
	} deleted;

	struct 
	{
		zMacInfo_s 		info;
	} macNS;

	struct 
	{
		zUnixInfo_s 	info;
		LONG				offsetToData;
	} unixNS;

	zVolumeInfo_s		vol;
	zPoolInfo_s			pool;
	LONG					extAttrUserFlags;
	BYTE					variableData[zGET_INFO_VARIABLE_DATA_SIZE];

} zInfo_s;

RCODE DfsMapError(
	LONG					lResult,
	RCODE					defaultRc);

LONG FlaimToNDSOpenFlags(
   FLMUINT				uiAccess,
	FLMBOOL				bDoDirectIo);

FLMUINT FlaimToNSSOpenFlags(
   FLMUINT				uiAccess,
	FLMBOOL				bDoDirectIo);

typedef FLMINT (* zROOT_KEY_FUNC)(
	FLMUINT				connectionID,
	FLMINT64 *			retRootKey);

typedef FLMINT (* zCLOSE_FUNC)(
	FLMINT64				key);

typedef FLMINT (* zCREATE_FUNC)(
	FLMINT64				key,
	FLMUINT				taskID,	
	FLMUINT64			xid,
	FLMUINT				nameSpace,
	const void *		path,
	FLMUINT				fileType,
	FLMUINT64			fileAttributes,
	FLMUINT				createFlags,
	FLMUINT				requestedRights,
	FLMINT64 *			retKey);

typedef FLMINT (* zOPEN_FUNC)(
	FLMINT64				key,
	FLMUINT				taskID,
	FLMUINT				nameSpace,
	const void *		path,
	FLMUINT				requestedRights,
	FLMINT64 *			retKey);

typedef FLMINT (* zDELETE_FUNC)(
	FLMINT64				key,
	FLMUINT64			xid,
	FLMUINT				nameSapce,
	const void *		path,
	FLMUINT				match,
	FLMUINT				deleteFlags);

typedef FLMINT (* zREAD_FUNC)(
	FLMINT64				key,
	FLMUINT64			xid,	
	FLMUINT64			startingOffset,
	FLMUINT				bytesToRead,
	void *				retBuffer,
	FLMUINT *			retBytesRead);

typedef FLMINT (* zDIO_READ_FUNC)(
	FLMINT64				key,
	FLMUINT64			unitOffset,
	FLMUINT				unitsToRead,
	FLMUINT				callbackData,
	void					(*dioReadCallBack)(
								FLMUINT	reserved,
								FLMUINT	callbackData,
								FLMUINT 	retStatus),
	void *				retBuffer);

typedef FLMINT (* zGET_INFO_FUNC)(
	FLMINT64				key,
	FLMUINT64			getInfoMask,
	FLMUINT				sizeRetGetInfo,
	FLMUINT				infoVersion,
	zInfo_s *			retGetInfo);

typedef FLMINT (* zMODIFY_INFO_FUNC)(
	FLMINT64				key,
	FLMUINT64			xid,
	FLMUINT64			modifyInfoMask,
	FLMUINT				sizeModifyInfo,
	FLMUINT				infoVersion,
	const zInfo_s *	modifyInfo);

typedef FLMINT (* zSET_EOF_FUNC)(
	FLMINT64				key,
	FLMUINT64			xid,	
	FLMUINT64			startingOffset,
	FLMUINT				flags);

typedef FLMINT (* zWRITE_FUNC)(
	FLMINT64				key,
	FLMUINT64			xid,	
	FLMUINT64			startingOffset,
	FLMUINT				bytesToWrite,
	const void *		buffer,
	FLMUINT *			retBytesWritten);

typedef FLMINT (* zDIO_WRITE_FUNC)(
	FLMINT64				key,
	FLMUINT64			unitOffset,
	FLMUINT				unitsToWrite,
	FLMUINT				callbackData,
	void					(*dioWriteCallBack)(
								FLMUINT	reserved,
								FLMUINT	callbackData,
								FLMUINT	retStatus),
	const void *		buffer);

typedef FLMINT (* zRENAME_FUNC)(
	FLMINT64				key,
	FLMUINT64			xid,
	FLMUINT				srcNameSpace,
	const void *		srcPath,
	FLMUINT				srcMatchAttributes,
	FLMUINT				dstNameSpace,
	const void *		dstPath,
	FLMUINT				renameFlags);

typedef BOOL (* zIS_NSS_VOLUME_FUNC)(
	const char *		path);

FSTATIC zIS_NSS_VOLUME_FUNC		gv_zIsNSSVolumeFunc = NULL;
FSTATIC zROOT_KEY_FUNC				gv_zRootKeyFunc = NULL;
FSTATIC zCLOSE_FUNC					gv_zCloseFunc = NULL;
FSTATIC zCREATE_FUNC					gv_zCreateFunc = NULL;
FSTATIC zOPEN_FUNC					gv_zOpenFunc = NULL;
FSTATIC zDELETE_FUNC					gv_zDeleteFunc = NULL;
FSTATIC zREAD_FUNC					gv_zReadFunc = NULL;
FSTATIC zDIO_READ_FUNC				gv_zDIOReadFunc = NULL;
FSTATIC zGET_INFO_FUNC				gv_zGetInfoFunc = NULL;
FSTATIC zMODIFY_INFO_FUNC			gv_zModifyInfoFunc = NULL;
FSTATIC zSET_EOF_FUNC				gv_zSetEOFFunc = NULL;
FSTATIC zWRITE_FUNC					gv_zWriteFunc = NULL;
FSTATIC zDIO_WRITE_FUNC				gv_zDIOWriteFunc = NULL;
FSTATIC zRENAME_FUNC					gv_zRenameFunc = NULL;
extern RCODE 							gv_CriticalFSError;

FSTATIC void ConvertToQualifiedNWPath(
	const char *		pInputPath,
	char *				pQualifiedPath);

FSTATIC RCODE nssTurnOffRenameInhibit(
	const char *		pszFileName);

FSTATIC LONG ConvertPathToLNameFormat(
	const char *		pPath,
	LONG *				plVolumeID,
	FLMBOOL *			pbNssVolume,
	FLMBYTE *			pLNamePath,
	LONG *				plLNamePathCount);

FSTATIC void DirectIONoWaitCallBack(
	LONG					unknownAlwaysZero,
	LONG					callbackData,
	LONG 					completionCode);

FSTATIC void nssDioCallback(
	FLMUINT				reserved,
	FLMUINT				UserData,
	FLMUINT				retStatus);

FSTATIC RCODE MapNSSToFlaimError(
	FLMINT				lStatus,
	RCODE					defaultRc);

FLMINT64		gv_NssRootKey;
FLMBOOL		gv_bNSSKeyInitialized = FALSE;

/***************************************************************************
Desc:	Initialize the root NSS key.
***************************************************************************/
RCODE nssInitialize( void)
{
	RCODE		rc = FERR_OK;
	FLMINT	lStatus;

	if (!gv_bNSSKeyInitialized)
	{
		// Import the required NSS functions

		if( (gv_zIsNSSVolumeFunc = (zIS_NSS_VOLUME_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x0C" "zIsNSSVolume")) == NULL)
		{
			// NSS is not available on this server.  Jump to exit.
			goto Exit;
		}
		
		if( (gv_zRootKeyFunc = (zROOT_KEY_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x08" "zRootKey")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zCloseFunc = (zCLOSE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x06" "zClose")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zCreateFunc = (zCREATE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zCreate")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zOpenFunc = (zOPEN_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x05" "zOpen")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zDeleteFunc = (zDELETE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zDelete")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zReadFunc = (zREAD_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x05" "zRead")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zDIOReadFunc = (zDIO_READ_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x08" "zDIORead")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zGetInfoFunc = (zGET_INFO_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x08" "zGetInfo")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zModifyInfoFunc = (zMODIFY_INFO_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x0B" "zModifyInfo")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zSetEOFFunc = (zSET_EOF_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zSetEOF")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zWriteFunc = (zWRITE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x06" "zWrite")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		
		if( (gv_zDIOWriteFunc = (zDIO_WRITE_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x09" "zDIOWrite")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		if( (gv_zRenameFunc = (zRENAME_FUNC)ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x07" "zRename")) == NULL)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}

		// Get the NSS root key

		if ((lStatus = gv_zRootKeyFunc( 0, &gv_NssRootKey)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_INITIALIZING_IO_SYSTEM);
			goto Exit;
		}
		gv_bNSSKeyInitialized = TRUE;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Close the root NSS key.
***************************************************************************/
void nssUninitialize( void)
{
	if (gv_bNSSKeyInitialized)
	{
		(void)gv_zCloseFunc( gv_NssRootKey);
		gv_bNSSKeyInitialized = FALSE;
	}
}

/***************************************************************************
Desc:	Maps NDS errors to IO errors.
		fix: The set of return codes returned by the NDS I/O functions (like
		NDSOpenStreamFile) is defined in "nw500\errors.h"  Unfortunately,
		some of the names used for the return codes are used in other include
		files.  Because of the conflict, it has been decided to use the
		integers associated with the names, rather than the names
		themselves.  This needs to be fixed.
***************************************************************************/
RCODE MapNWtoFlaimError(
	LONG		lResult,
	RCODE		defaultRc
	)
{
	RCODE		rc;
	switch (lResult)
	{
		case 128: // ERR_LOCK_FAIL
		case 147: // ERR_NO_READ_PRIVILEGE
		case 148: // ERR_NO_WRITE_PRIVILEGE
		case 168: // ERR_ACCESS_DENIED
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			break;

		case 136: //ERR_INVALID_FILE_HANDLE
			rc = RC_SET( FERR_IO_BAD_FILE_HANDLE);
			break;

		case 001: //ERR_INSUFFICIENT_SPACE
		case 153: //ERR_DIRECTORY_FULL
			rc = RC_SET( FERR_IO_DISK_FULL);
			break;

		case 130: //ERR_NO_OPEN_PRIVILEGE
		case 165: //ERR_INVALID_OPENCREATE_MODE
			rc = RC_SET( FERR_IO_OPEN_ERR);
			break;

		case 158: //ERR_BAD_FILE_NAME
			rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
			break;

		case 129: //ERR_OUT_OF_HANDLES
			rc = RC_SET( FERR_IO_TOO_MANY_OPEN_FILES);
			break;

		case 139: //ERR_NO_RENAME_PRIVILEGE
		case 154: //ERR_RENAME_ACROSS_VOLUME
		case 164: //ERR_RENAME_DIR_INVALID
			rc = RC_SET( FERR_IO_RENAME_FAILURE);
			break;

		case 222: //ERR_BAD_PASSWORD
		case 223: //ERR_PASSWORD_EXPIRED
			rc = RC_SET( FERR_IO_INVALID_PASSWORD);
			break;

		case 156: //ERR_INVALID_PATH
			rc = RC_SET( FERR_IO_INVALID_PATH);
			break;

		case 122: //ERR_CONNECTION_ALREADY_TEMPORARY
		case 123: //ERR_CONNECTION_ALREADY_LOGGED_IN
		case 124: //ERR_CONNECTION_NOT_AUTHENTICATED
		case 125: //ERR_CONNECTION_NOT_LOGGED_IN
		case 224: //ERR_NO_LOGIN_CONNECTIONS_AVAILABLE
			rc = RC_SET( FERR_IO_CONNECT_ERROR);
			break;

		default:
			rc = RC_SET( defaultRc);
			break;
	}
	return( rc );
}

/***************************************************************************
Desc:	Maps NSS errors to IO errors.
***************************************************************************/
FSTATIC RCODE MapNSSToFlaimError(
	FLMINT	lStatus,
	RCODE		defaultRc)
{
	RCODE		rc;

	switch (lStatus)
	{
		case zERR_FILE_ALREADY_EXISTS:
		case zERR_DIRECTORY_NOT_EMPTY:
		case zERR_DIR_CANNOT_BE_OPENED:
		case zERR_NO_SET_PRIVILEGE:
		case zERR_NO_CREATE_PRIVILEGE:
		case zERR_ACCESS_DENIED:
		case zERR_NO_WRITE_PRIVILEGE:
		case zERR_NO_READ_PRIVILEGE:
		case zERR_NO_DELETE_PRIVILEGE:
		case zERR_SOME_NO_DELETE_PRIVILEGE:
		case zERR_CANT_DELETE_OPEN_FILE:
		case zERR_NO_CREATE_DELETE_PRIVILEGE:
		case zERR_NO_SALVAGE_PRIVILEGE:
		case zERR_FILE_READ_LOCKED:
		case zERR_FILE_WRITE_LOCKED:
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			break;

		case zERR_BAD_FILE_HANDLE:
			rc = RC_SET( FERR_IO_BAD_FILE_HANDLE);
			break;

		case zERR_OUT_OF_SPACE:
			rc = RC_SET( FERR_IO_DISK_FULL);
			break;

		case zERR_NO_OPEN_PRIVILEGE:
			rc = RC_SET( FERR_IO_OPEN_ERR);
			break;

		case zERR_NAME_NOT_FOUND_IN_DIRECTORY:
		case zERR_NO_FILES_FOUND:
		case zERR_VOLUME_NOT_FOUND:
		case zERR_NO_SUCH_OBJECT:
			rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
			break;

		case zERR_NO_MORE_CONTEXT_HANDLE_IDS:
			rc = RC_SET( FERR_IO_TOO_MANY_OPEN_FILES);
			break;
		case zERR_ALL_FILES_IN_USE:
		case zERR_SOME_FILES_IN_USE:
		case zERR_ALL_FILES_READ_ONLY:
		case zERR_SOME_FILES_READ_ONLY:
		case zERR_ALL_NAMES_EXIST:
		case zERR_SOME_NAMES_EXIST:
		case zERR_NO_RENAME_PRIVILEGE:
		case zERR_RENAME_DIR_INVALID:
		case zERR_RENAME_TO_OTHER_VOLUME:
		case zERR_CANT_RENAME_DATA_STREAMS:
		case zERR_FILE_RENAME_IN_PROGRESS:
		case zERR_CANT_RENAME_TO_DELETED:
			rc = RC_SET( FERR_IO_RENAME_FAILURE);
			break;

		case zERR_INVALID_NAME:
		case zERR_INVALID_CHAR_IN_NAME:
		case zERR_INVALID_PATH:
		case zERR_NO_NAMES_IN_PATH:
		case zERR_NO_MORE_NAMES_IN_PATH:
		case zERR_PATH_MUST_BE_FULLY_QUALIFIED:
		case zERR_NAME_NO_LONGER_VALID:
		case zERR_INVALID_PATH_FORMAT:
		case zERR_INVALID_PATH_SEPARATOR:
		case zERR_VOLUME_SEPARATOR_NOT_SUPPORTED:
		case zERR_BAD_VOLUME_NAME:
			rc = RC_SET( FERR_IO_INVALID_PATH);
			break;
		case zERR_CONNECTION_NOT_LOGGED_IN:
			rc = RC_SET( FERR_IO_CONNECT_ERROR);
			break;
		case zERR_NO_MEMORY:
			rc = RC_SET( FERR_MEM);
			break;
		case zERR_NOT_SUPPORTED:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
		case zERR_END_OF_FILE:
		case zERR_BEYOND_EOF:
			rc = RC_SET( FERR_IO_END_OF_FILE);
			break;

		default:
			rc = RC_SET( defaultRc);
			break;
	}
	return( rc );
}

/***************************************************************************
Desc:	Maps direct IO errors to IO errors.
fix: we shouldn't have 2 copies of this function.  this is just temporary.
      long term, we need to make the FDFS.CPP version public.
***************************************************************************/
RCODE DfsMapError(
	LONG	lResult,
	RCODE	defaultRc
	)
{
	switch (lResult)
	{
		case DFSHoleInFileError:
		case DFSOperationBeyondEndOfFile:
			return( RC_SET( FERR_IO_END_OF_FILE));
		case DFSHardIOError:
		case DFSInvalidFileHandle:
			return( RC_SET( FERR_IO_BAD_FILE_HANDLE));
		case DFSNoReadPrivilege:
			return( RC_SET( FERR_IO_ACCESS_DENIED));
		case DFSInsufficientMemory:
			return( RC_SET( FERR_MEM));
		default:
			return( RC_SET( defaultRc));
	}
}


/****************************************************************************
Desc:		Map flaim I/O flags to NDS I/O flags
****************************************************************************/
LONG FlaimToNDSOpenFlags(
   FLMUINT	uiAccess,
	FLMBOOL	bDoDirectIo
	)
{
	LONG	lFlags = NO_RIGHTS_CHECK_ON_OPEN_BIT | ALLOW_SECURE_DIRECTORY_ACCESS_BIT;

	if (uiAccess & (F_IO_RDONLY | F_IO_RDWR))
	{
		lFlags |= READ_ACCESS_BIT;
	}
	if (uiAccess & F_IO_RDWR)
	{
		lFlags |= WRITE_ACCESS_BIT;
	}

	if (uiAccess & F_IO_SH_DENYRW )
	{
		lFlags |= DENY_READ_BIT;
	}
	if (uiAccess & (F_IO_SH_DENYWR | F_IO_SH_DENYRW))
	{
		lFlags |= DENY_WRITE_BIT;
	}
	if (bDoDirectIo)
	{
		lFlags |= NEVER_READ_AHEAD_BIT;
	}
	return( lFlags );
}

/****************************************************************************
Desc:		Map flaim I/O flags to NDS I/O flags for NSS volumes
****************************************************************************/
FLMUINT FlaimToNSSOpenFlags(
   FLMUINT	uiAccess,
	FLMBOOL	bDoDirectIo)
{
	FLMUINT	lFlags = zRR_ALLOW_SECURE_DIRECTORY_ACCESS |
						zRR_CANT_DELETE_WHILE_OPEN;

	if (uiAccess & (F_IO_RDONLY | F_IO_RDWR))
	{
		lFlags |= zRR_READ_ACCESS;
	}
	if (uiAccess & F_IO_RDWR)
	{
		lFlags |= zRR_WRITE_ACCESS;
	}

	if (uiAccess & F_IO_SH_DENYRW)
	{
		lFlags |= zRR_DENY_READ;
	}
	if (uiAccess & (F_IO_SH_DENYWR | F_IO_SH_DENYRW))
	{
		lFlags |= zRR_DENY_WRITE;
	}
	if (bDoDirectIo)
	{
		lFlags |= zRR_DIO_MODE;
	}
	return( lFlags );
}

/****************************************************************************
Desc:		Map flaim I/O flags to NetWare I/O flags
****************************************************************************/
LONG FlaimToNWOpenFlags(
   FLMUINT		uiAccess,
	FLMBOOL		bDoDirectIo
	)
{
	LONG	lFlags = 0;

	if (uiAccess & (F_IO_RDONLY | F_IO_RDWR))
	{
		lFlags |= READ_ACCESS_BIT;
	}
	if (uiAccess & F_IO_RDWR)
	{
		lFlags |= WRITE_ACCESS_BIT;
	}

	if (uiAccess & F_IO_SH_DENYRW )
	{
		lFlags |= DENY_READ_BIT;
	}
	if (uiAccess & (F_IO_SH_DENYWR | F_IO_SH_DENYRW))
	{
		lFlags |= DENY_WRITE_BIT;
	}
	if (bDoDirectIo)
	{
		lFlags |= NEVER_READ_AHEAD_BIT;
	}
	return( lFlags );
}

/****************************************************************************
Desc:		Default Constructor for F_FileHdl class
****************************************************************************/
F_FileHdlImp::F_FileHdlImp()
{
	// Should call the base class constructor automatically.

	m_lFileHandle = -1;
	m_lOpenAttr = 0;
	m_uiCurrentPos = 0;
	m_lVolumeID = F_NW_DEFAULT_VOLUME_NUMBER;
	m_bDoSuballocation = FALSE;
	m_lLNamePathCount = 0;
	m_pszIoPath = NULL;
	m_uiExtendSize = 0;
	m_uiMaxAutoExtendSize = gv_FlmSysData.uiMaxFileSize;

	// Direct IO members
	m_bDoDirectIO = FALSE;	// TRUE = do direct IO-style read/writes
	m_lSectorsPerBlock = 0;
	m_lMaxBlocks = 0;

	m_bNSS = FALSE;
	m_bNSSFileOpen = FALSE;
}

/***************************************************************************
Desc:		Open or create a file.
***************************************************************************/
RCODE F_FileHdlImp::OpenOrCreate(
	const char *		pFileName,
   FLMUINT				uiAccess,
	FLMBOOL				bCreateFlag)
{
	RCODE					rc = FERR_OK;
	LONG					unused;
	void *				unused2;
	char *				pszQualifiedPath = NULL;
	LONG					lErrorCode;
	FLMBYTE *			pTmpLNamePath;
	char *				pSaveFileName;
	FLMBYTE *			pLNamePath;
	LONG *				plLNamePathCount;
	LONG					LNamePathCount;
	struct VolumeInformationStructure *
							pVolumeInfo;
	char *				pszTemp;
	char *				pIoDirPath;
	FLMBOOL				bNssVolume = FALSE;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	m_bDoDirectIO = (uiAccess & F_IO_DIRECT) ? TRUE : FALSE;

	if( uiAccess & F_IO_DELETE_ON_CLOSE)
	{
		if( !m_pszIoPath)
		{
			if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszIoPath)))
			{
				goto Exit;
			}
		}
		f_strcpy( m_pszIoPath, pFileName);
		m_bDeleteOnClose = TRUE;
	}
	else
	{
		m_bDeleteOnClose = FALSE;
	}

	if (RC_BAD( rc = f_alloc(
								(FLMUINT)(F_PATH_MAX_SIZE +
											 F_PATH_MAX_SIZE +
											 F_PATH_MAX_SIZE +
											 F_PATH_MAX_SIZE +
											 sizeof( struct VolumeInformationStructure) +
											 F_PATH_MAX_SIZE),
											 &pszQualifiedPath)))
	{
		goto Exit;
	}

	pTmpLNamePath = (((FLMBYTE *)pszQualifiedPath) + F_PATH_MAX_SIZE);
	pSaveFileName = (((char *)pTmpLNamePath) + F_PATH_MAX_SIZE);
	pIoDirPath = (((char *)pSaveFileName) + F_PATH_MAX_SIZE);
	pVolumeInfo = (struct VolumeInformationStructure *)
								(((char *)pIoDirPath) + F_PATH_MAX_SIZE);
	pszTemp = (char *)(((char *)(pVolumeInfo)) +
									sizeof( struct VolumeInformationStructure));


	/* Save the file name in case we have to create the directory. */

	if((bCreateFlag) && (uiAccess & F_IO_CREATE_DIR))
	{
		f_strcpy( pSaveFileName, pFileName);
	}

	if( !m_pszIoPath)
	{
		pLNamePath = pTmpLNamePath;
		plLNamePathCount = &LNamePathCount;
	}
	else
	{
		pLNamePath = (FLMBYTE *)m_pszIoPath;
		plLNamePathCount = &m_lLNamePathCount;
	}
	
	ConvertToQualifiedNWPath( pFileName, pszQualifiedPath);

	lErrorCode = ConvertPathToLNameFormat(
		pszQualifiedPath,
		&m_lVolumeID,
		&bNssVolume,
		pLNamePath,
		plLNamePathCount);
	if( lErrorCode != 0)
	{
      rc = MapNWtoFlaimError( lErrorCode, FERR_PARSING_FILE_NAME);
		goto Exit;
   }

	// Determine if the volume is NSS or not

	if (gv_bNSSKeyInitialized && bNssVolume)
	{
		m_bNSS = TRUE;
	}

	if ( m_bDoDirectIO )
	{
		if (!m_bNSS)
		{
			lErrorCode = 
				ReturnVolumeMappingInformation( m_lVolumeID, pVolumeInfo);
			if (lErrorCode != 0)
			{
				rc = DfsMapError( lErrorCode, FERR_INITIALIZING_IO_SYSTEM);
				goto Exit;
			}
			
			m_lSectorsPerBlock = (LONG)(pVolumeInfo->VolumeAllocationUnitSizeInBytes /
										NLM_SECTOR_SIZE);
			m_lMaxBlocks = (LONG)(gv_FlmSysData.uiMaxFileSize /
								(FLMUINT)pVolumeInfo->VolumeAllocationUnitSizeInBytes);
		}
		else
		{
			m_lMaxBlocks = (LONG)(gv_FlmSysData.uiMaxFileSize / (FLMUINT)65536);
		}
	}

	/*
	Set up the file characteristics requested by caller.
	*/

	if (bCreateFlag)
	{

		// File is to be created

		if (NWTestIfFileExists( pszQualifiedPath ) == FERR_OK)
		{
			if (uiAccess & F_IO_EXCL)
			{
				rc = RC_SET( FERR_IO_ACCESS_DENIED);
				goto Exit;
			}
			
			(void)NWDeleteFile( pszQualifiedPath);
		}
	}

	/* Try to create or open the file */

	if (m_bNSS)
	{
		FLMINT	lStatus;

		if (bCreateFlag)
		{
			FLMUINT64	qFileAttr;

			qFileAttr = (FLMUINT64)(((m_bDoSuballocation)
									 ? (FLMUINT)(zFA_DO_NOT_COMPRESS_FILE)
									 : (FLMUINT)(zFA_NO_SUBALLOC |
												 zFA_DO_NOT_COMPRESS_FILE)) |
									zFA_IMMEDIATE_PURGE);

Retry_NSS_Create:

			m_lOpenAttr = FlaimToNSSOpenFlags( uiAccess, m_bDoDirectIO);
			if ((lStatus = gv_zCreateFunc( gv_NssRootKey, 1, 0,
				zNSPACE_LONG | zMODE_UTF8, pszQualifiedPath, zFILE_REGULAR,
				qFileAttr, zCREATE_DELETE_IF_THERE, (FLMUINT)m_lOpenAttr,
				&m_NssKey)) != zOK)
			{
				if (uiAccess & F_IO_CREATE_DIR)
				{
					uiAccess &= ~F_IO_CREATE_DIR;

					// Remove the file name for which we are creating the directory.

					if( f_pathReduce( pSaveFileName, pIoDirPath,	pszTemp) == FERR_OK)
					{
						F_FileSystemImp	FileSystem;

						if (RC_OK( FileSystem.CreateDir( pIoDirPath)))
							goto Retry_NSS_Create;
					}
				}
				rc = MapNSSToFlaimError( lStatus,
							(RCODE)(m_bDoDirectIO
									  ? (RCODE)FERR_DIRECT_CREATING_FILE
									  : (RCODE)FERR_CREATING_FILE));
				goto Exit;
			}
		}
		else
		{
			m_lOpenAttr = FlaimToNSSOpenFlags( uiAccess, m_bDoDirectIO);
			if ((lStatus = gv_zOpenFunc(
							gv_NssRootKey,					// FLMINT64			key
							1,									// FLMUINT			taskID
							zNSPACE_LONG | zMODE_UTF8,	// FLMUINT			nameSpace
							pszQualifiedPath,				// const void *	path
							(FLMUINT)m_lOpenAttr,		// FLMUINT			requestedRights
							&m_NssKey)) != zOK)
			{
				rc = MapNSSToFlaimError( lStatus,
							(RCODE)(m_bDoDirectIO
									  ? (RCODE)FERR_DIRECT_OPENING_FILE
									  : (RCODE)FERR_OPENING_FILE));
				goto Exit;
			}
		}
		m_bNSSFileOpen = TRUE;
	}
	else
	{
		if (bCreateFlag)
		{
			m_lOpenAttr = (LONG)(((m_bDoSuballocation)
											 ? (LONG)(DO_NOT_COMPRESS_FILE_BIT)
											 : (LONG)(NO_SUBALLOC_BIT |
														 DO_NOT_COMPRESS_FILE_BIT)) | 
										IMMEDIATE_PURGE_BIT);

Retry_Create:
			lErrorCode = CreateFile(
				0,						// Connection ID
				1,						// Task ID
				m_lVolumeID,		// Volume ID
				0,						// (path base) 0 because we supply fully-qualified path
				(BYTE *)pLNamePath,// LNAME-array format path
				*plLNamePathCount,	// LNAME-array size
				LONGNameSpace,			// LONGNameSpace
				m_lOpenAttr,
				0xff,					// 0xff, 0, DELETE_FILE_ON_CREATE_BIT, READ_ACCESS_BIT|WRITE_ACCESS_BIT
				PrimaryDataStream,	// PrimaryDataStream
				&m_lFileHandle,
				&unused,				// return the directory number
				&unused2				// 	union DirUnion *Entry;
				);

			if ((lErrorCode != 0) && (uiAccess & F_IO_CREATE_DIR))
			{
				uiAccess &= ~F_IO_CREATE_DIR;

				/* Remove the file name for which we are creating the directory. */

				if( f_pathReduce( pSaveFileName, pIoDirPath,	pszTemp) == FERR_OK)
				{
					F_FileSystemImp	FileSystem;

					if (RC_OK( FileSystem.CreateDir( pIoDirPath)))
						goto Retry_Create;
				}
			}

			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( FERR_IO_ACCESS_DENIED);
				goto Exit;
			}
		}
		else
		{
			m_lOpenAttr = FlaimToNWOpenFlags(uiAccess, m_bDoDirectIO);
			lErrorCode = OpenFile(
				0,							// Connection ID
				1,							// Task ID
				m_lVolumeID,			// Volume ID
				0,							// (path base) 0 because we supply fully-qualified path
				(BYTE *)pLNamePath,	// LNAME-array format path
				*plLNamePathCount,	// LNAME-array size
				LONGNameSpace,			// LONGNameSpace
				0,							// ?? LONG MatchBits,
				m_lOpenAttr,
				PrimaryDataStream,
				&m_lFileHandle,
				&unused,					// return the directory number
				&unused2					// union DirUnion *Entry;
				);

			/*
			Too many error codes map to 255, so we put in a special
			case check here.
			*/

			if( lErrorCode == 255)
			{
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
				goto Exit;
			}
		}

		/* Check if the file operation was successful */

		if ( lErrorCode != 0)
		{
			rc = MapNWtoFlaimError( lErrorCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)(m_bDoDirectIO
												? (RCODE)FERR_DIRECT_CREATING_FILE
												: (RCODE)FERR_CREATING_FILE)
								  : (RCODE)(m_bDoDirectIO
												? (RCODE)FERR_DIRECT_OPENING_FILE
												: (RCODE)FERR_OPENING_FILE)));
			goto Exit;
		}

		if (bCreateFlag)
		{
			/* Close and reopen the file after creating it. */

			// Revoke the file handle rights and close the file
			// (signified by passing 2 for the QueryFlag parameter).
			// If this call fails and returns a 255 error, it may
			// indicate that the FILESYS.NLM being used on the server
			// does not implement option 2 for the QueryFlag parameter.
			// In this case, we will default to our old behavior
			// and simply call CloseFile.  This, potentially, will
			// not free all of the lock objects and could result in
			// a memory leak in filesys.nlm.  However, we want to
			// at least make sure that there is a corresponding
			// RevokeFileHandleRights or CloseFile call for every
			// file open / create call.

			if( (lErrorCode = RevokeFileHandleRights( 0, 1, 
					m_lFileHandle, 2, m_lOpenAttr & 0x0000000F, &unused)) == 0xFF)
			{
				lErrorCode = CloseFile( 0, 1, m_lFileHandle);
			}
			m_lOpenAttr = 0;

			if ( lErrorCode != 0 )
			{
				rc = MapNWtoFlaimError(lErrorCode, FERR_CLOSING_FILE);
				goto Exit;
			}

			m_lOpenAttr = FlaimToNWOpenFlags(uiAccess, m_bDoDirectIO);
			lErrorCode = OpenFile(			/* moved to vswitch.386 */
				0,						// Connection ID
				1,						// Task ID
				m_lVolumeID,		// Volume ID
				0,						// (path base) 0 because we supply fully-qualified path
				(BYTE *)pLNamePath,// LNAME-array format path
				*plLNamePathCount,// LNAME-array size
				LONGNameSpace,		// LONGNameSpace
				0,						// ?? LONG MatchBits,
				m_lOpenAttr,
				PrimaryDataStream,
				&m_lFileHandle,
				&unused,				// return the directory number
				&unused2				// union DirUnion *Entry;
				);

			if ( lErrorCode != 0 )
			{
				/*
				Too many error codes map to 255, so we put in a special
				case check here.
				*/

				if( lErrorCode == 255)
				{
					rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
				}
				else
				{
					rc = MapNWtoFlaimError( lErrorCode,
								(RCODE)(m_bDoDirectIO
										 ? (RCODE)FERR_DIRECT_OPENING_FILE
										 : (RCODE)FERR_OPENING_FILE));
				}
				goto Exit;
			}
		}
		
		if ( m_bDoDirectIO )
		{
			lErrorCode = SwitchToDirectFileMode(0, m_lFileHandle);
			if (lErrorCode != 0)
			{
				if (RevokeFileHandleRights( 0, 1, 
					m_lFileHandle, 2, m_lOpenAttr & 0x0000000F, &unused) == 0xFF)
				{
					(void)CloseFile( 0, 1, m_lFileHandle);
				}
				rc = MapNWtoFlaimError( lErrorCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)FERR_DIRECT_CREATING_FILE
								  : (RCODE)FERR_DIRECT_OPENING_FILE));
				goto Exit;
			}
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		m_lFileHandle = -1;
		m_lOpenAttr = 0;
		m_bNSSFileOpen = FALSE;
	}

	if (pszQualifiedPath)
	{
		f_free( &pszQualifiedPath);
	}

   return( rc );
}

/****************************************************************************
Desc:		Create a file 
****************************************************************************/
RCODE F_FileHdlImp::Create(
	const char *		pIoPath,
	FLMUINT				uiIoFlags)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bFileOpened == FALSE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = OpenOrCreate( pIoPath, uiIoFlags, TRUE)))
	{
		goto Exit;
	}

	m_bFileOpened = TRUE;
	m_uiCurrentPos = 0;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlImp::CreateUnique(
	char *				pIoPath,
	const char *		pszFileExtension,
	FLMUINT				uiIoFlags)
{
	RCODE			rc = FERR_OK;
	char *		pszTmp;
	FLMBOOL		bModext = TRUE;
	FLMUINT		uiBaseTime = 0;
	char			ucHighByte = 0;
	char			ucFileName[ F_FILENAME_SIZE];
	char			szDirPath[ F_PATH_MAX_SIZE];
	char			szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT		uiCount;

	f_memset( ucFileName, 0, sizeof( ucFileName));

	flmAssert( m_bFileOpened == FALSE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	f_strcpy( szDirPath, pIoPath);

   // Search backwards replacing trailing spaces with NULLs.

	pszTmp = szDirPath;
	pszTmp += (f_strlen( pszTmp) - 1);
	while ((*pszTmp == 0x20) && pszTmp >= szDirPath)
	{
		*pszTmp = 0;
		pszTmp--;
	}

	/* Append a backslash if one isn't already there. */

	if (pszTmp >= szDirPath && *pszTmp != '\\')
	{
		pszTmp++;
		*pszTmp++ = '\\';
	}
	else
	{
		pszTmp++;
	}
	*pszTmp = 0;

	if ((pszFileExtension) && (f_strlen( pszFileExtension) >= 3))
	{
		bModext = FALSE;
	}

	uiCount = 0;
	do
	{
		f_pathCreateUniqueName( &uiBaseTime,  ucFileName, pszFileExtension,
										&ucHighByte, bModext);

		f_strcpy( szTmpPath, szDirPath);
		f_pathAppend( szTmpPath, ucFileName);

		rc = Create( szTmpPath, uiIoFlags | F_IO_EXCL);

		if (rc == FERR_IO_DISK_FULL)
		{
			(void)NWDeleteFile( szTmpPath);
			goto Exit;
		}

		if ((rc == FERR_IO_PATH_NOT_FOUND) || (rc == FERR_IO_INVALID_PASSWORD))
		{
			goto Exit;
		}
	} while ((rc != FERR_OK) && (uiCount++ < F_IO_MAX_CREATION_TRIES));

   // Check if the path was created

   if ((uiCount >= F_IO_MAX_CREATION_TRIES) && (rc != FERR_OK))
   {
		rc = RC_SET( FERR_IO_PATH_CREATE_FAILURE);
		goto Exit;
   }

	m_bFileOpened = TRUE;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

	// Created file name needs to be returned.

	f_strcpy( pIoPath, szTmpPath);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Open a file
****************************************************************************/
RCODE F_FileHdlImp::Open(
	const char *	pIoPath,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiStartTime;
	FLMUINT			ui15Secs;
	FLMUINT			uiCurrTime;

	flmAssert( m_bFileOpened == FALSE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	// Loop on error open conditions.
	
	FLM_SECS_TO_TIMER_UNITS(15 /*seconds*/, ui15Secs);
	uiStartTime = FLM_GET_TIMER();

	do
	{
		if( RC_OK( rc = OpenOrCreate( pIoPath, uiIoFlags, FALSE)))
			break;

		if( rc != FERR_IO_TOO_MANY_OPEN_FILES)
		{
			goto Exit;
		}

		// If for some reason we cannot open the file, then
		// try to close some other file handle in the list.

		if( RC_BAD( gv_FlmSysData.pFileHdlMgr->ReleaseOneAvail()))
		{
			goto Exit;
		}

		f_sleep( 50);
		uiCurrTime = FLM_GET_TIMER();
	} while( FLM_ELAPSED_TIME( uiCurrTime, uiStartTime) < ui15Secs);
	
	if ( RC_BAD(rc) )
	{
		goto Exit;
	}

	m_bFileOpened = TRUE;
	m_uiCurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & F_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Close a file
****************************************************************************/
RCODE F_FileHdlImp::Close()
{
	LONG			unused;
	FLMBOOL		bDeleteAllowed = TRUE;
	RCODE			rc = FERR_OK;

	if( !m_bFileOpened)
	{
		goto Exit;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		bDeleteAllowed = FALSE;
	}

	if (m_bNSS)
	{
		if (m_bNSSFileOpen)
		{
			(void)gv_zCloseFunc( m_NssKey);
			m_bNSSFileOpen = FALSE;
		}
	}
	else
	{
		// Revoke the file handle rights and close the file
		// (signified by passing 2 for the QueryFlag parameter).
		// If this call fails and returns a 255 error, it may
		// indicate that the FILESYS.NLM being used on the server
		// does not implement option 2 for the QueryFlag parameter.
		// In this case, we will default to our old behavior
		// and simply call CloseFile.  This, potentially, will
		// not free all of the lock objects and could result in
		// a memory leak in filesys.nlm.  However, we want to
		// at least make sure that there is a corresponding
		// RevokeFileHandleRights or CloseFile call for every
		// file open / create call.

		if( RevokeFileHandleRights( 0, 1, 
				m_lFileHandle, 2, m_lOpenAttr & 0x0000000F, &unused) == 0xFF)
		{
			(void)CloseFile( 0, 1, m_lFileHandle);
		}
	}

	m_lOpenAttr = 0;
	m_lFileHandle = -1;
	m_bFileOpened = m_bOpenedReadOnly = m_bOpenedExclusive = FALSE;

	if( m_bDeleteOnClose)
	{
		if( bDeleteAllowed)
		{
			if (m_bNSS)
			{
				(void)gv_zDeleteFunc( gv_NssRootKey, 0, zNSPACE_LONG | zMODE_UTF8,
									m_pszIoPath, zMATCH_ALL, 0);
			}
			else
			{
				(void)EraseFile( 0, 1, m_lVolumeID, 0, (BYTE *)m_pszIoPath,
					m_lLNamePathCount, LONGNameSpace, 0);
			}
		}

		m_bDeleteOnClose = FALSE;
		m_lLNamePathCount = 0;
	}

	if( m_pszIoPath)
	{
		f_free( &m_pszIoPath);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Read from a file
****************************************************************************/
RCODE F_FileHdlImp::Read(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE		rc;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if ( m_bDoDirectIO )
	{
		rc = _DirectIORead( uiReadOffset, uiBytesToRead,	pvBuffer, puiBytesReadRV );
	}
	else
	{
		rc = _Read( uiReadOffset, uiBytesToRead,	pvBuffer, puiBytesReadRV );
	}

Exit:

	return( rc );
}


/****************************************************************************
Desc:		Read from a file
****************************************************************************/
RCODE F_FileHdlImp::_Read(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE			rc = FERR_OK;
	FCBType *	fcb;
	LONG			lBytesRead;
	LONG			lErr;
	
	flmAssert( m_bFileOpened == TRUE);

	if( puiBytesReadRV)
	{
		*puiBytesReadRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiReadOffset == F_IO_CURRENT_POS)
		uiReadOffset = m_uiCurrentPos;

	if (m_bNSS)
	{
		FLMINT	lStatus;
		FLMUINT	nBytesRead;

		if ((lStatus = gv_zReadFunc( m_NssKey, 0, (FLMUINT64)uiReadOffset,
									(FLMUINT)uiBytesToRead, pvBuffer,
									&nBytesRead)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_READING_FILE);
			goto Exit;
		}

		if( puiBytesReadRV)
		{
			*puiBytesReadRV = (FLMUINT)nBytesRead;
		}

		if ((FLMUINT)nBytesRead < uiBytesToRead)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
		}
		m_uiCurrentPos = uiReadOffset + (FLMUINT)nBytesRead;
	}
	else
	{
		lErr = MapFileHandleToFCB( m_lFileHandle, &fcb );
		if ( lErr != 0 )
		{
			rc = MapNWtoFlaimError( lErr, FERR_SETTING_UP_FOR_READ);
			goto Exit;
		}
		lErr = ReadFile( fcb->Station, m_lFileHandle, uiReadOffset,
					uiBytesToRead, &lBytesRead, pvBuffer);
		if ( lErr == 0 )
		{
			if( puiBytesReadRV)
			{
				*puiBytesReadRV = (FLMUINT) lBytesRead;
			}

			if (lBytesRead < (LONG)uiBytesToRead)
			{
				rc = RC_SET( FERR_IO_END_OF_FILE);
			}
			m_uiCurrentPos = uiReadOffset + lBytesRead;
		}
		else
		{
			rc = MapNWtoFlaimError(lErr, FERR_READING_FILE);
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Returns m_uiCurrentPos
****************************************************************************/
RCODE F_FileHdlImp::Tell(
	FLMUINT *	puiOffset)
{
	RCODE		rc;

	if( RC_OK( rc = GET_FS_ERROR()))
	{
		*puiOffset = m_uiCurrentPos;
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Calls the Direct IO-style read routine
Note:		Where possible, the caller should attempt to read on sector
			boundaries and full sectors.  This routine will do the
			necessary work if this is not done, but it will be less
			efficient.
****************************************************************************/
RCODE F_FileHdlImp::_DirectIORead(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;
	LONG			lStartSector;
	LONG			lSectorCount;
	LONG			lResult;
	BYTE			ucSectorBuf [NLM_SECTOR_SIZE];
	FLMUINT		uiBytesToCopy;
	FLMUINT		uiSectorOffset;
	FLMUINT		uiTotal;
	FLMINT		lStatus;

	flmAssert( m_bFileOpened == TRUE);

	if( puiBytesReadRV)
	{
		*puiBytesReadRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiReadOffset == F_IO_CURRENT_POS)
		uiReadOffset = m_uiCurrentPos;

	/* Calculate the starting sector. */

	lStartSector = uiReadOffset / NLM_SECTOR_SIZE;

	/*
	See if the offset is on a sector boundary.  If not, we must read
	into the local sector buffer and then copy into the buffer.
	We must also read into the local buffer if our read size is less
	than the sector size.
	*/

	if ((uiReadOffset % NLM_SECTOR_SIZE != 0) ||
		 (uiBytesToRead < NLM_SECTOR_SIZE))
	{
		if (m_bNSS)
		{
			if ((lStatus = gv_zDIOReadFunc( m_NssKey, 
									(FLMUINT64)lStartSector, 1,
									(FLMUINT)0, NULL, ucSectorBuf)) != zOK)
			{
				rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_READING_FILE);
				goto Exit;
			}
		}
		else
		{
			lResult = DirectReadFile(
				0,
				m_lFileHandle,
				lStartSector,
				1,
				ucSectorBuf
				);
			if (lResult != 0)
			{
				rc = DfsMapError( lResult, FERR_DIRECT_READING_FILE);
				goto Exit;
			}
		}

		/* Copy the part of the sector that was requested into the buffer. */

		uiSectorOffset = uiReadOffset % NLM_SECTOR_SIZE;

		if ((uiBytesToCopy = uiBytesToRead) > NLM_SECTOR_SIZE - uiSectorOffset)
			uiBytesToCopy = NLM_SECTOR_SIZE - uiSectorOffset;
		f_memcpy( pucBuffer, &ucSectorBuf [uiSectorOffset], uiBytesToCopy);
		pucBuffer += uiBytesToCopy;
		uiBytesToRead -= (FLMUINT)uiBytesToCopy;
		m_uiCurrentPos += (FLMUINT)uiBytesToCopy;

		if( puiBytesReadRV)
		{
   		(*puiBytesReadRV) += (FLMUINT)uiBytesToCopy;
		}

		/* See if we got everything we wanted to with this read. */

		if (!uiBytesToRead)
			goto Exit;

		/* Go to the next sector boundary */

		lStartSector++;
	}

	/*
	At this point, we are poised to read on a sector boundary.  See if we
	have at least one full sector to read.  If so, we can read it directly
	into the provided buffer.  If not, we must use the temporary sector
	buffer.
	*/

	if (uiBytesToRead >= NLM_SECTOR_SIZE)
	{
		lSectorCount = (LONG)(uiBytesToRead / NLM_SECTOR_SIZE);
Try_Read:
		if (m_bNSS)
		{
			if ((lStatus = gv_zDIOReadFunc( m_NssKey,
									(FLMUINT64)lStartSector,
									(FLMUINT)lSectorCount,
									(FLMUINT)0, NULL, pucBuffer)) != zOK)
			{
				if ((lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF) &&
					 (lSectorCount > 1))
				{

					// See if we can read one less sector.  We will return
					// FERR_IO_END_OF_FILE in this case.

					lSectorCount--;
					rc = RC_SET( FERR_IO_END_OF_FILE);
					goto Try_Read;
				}
				rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_READING_FILE);
				goto Exit;
			}
		}
		else
		{
			lResult = DirectReadFile(
				0,
				m_lFileHandle,
				lStartSector,
				lSectorCount,
				pucBuffer
				);
			if (lResult != 0)
			{
				if ((lResult == DFSOperationBeyondEndOfFile) &&
					 (lSectorCount > 1))
				{

					// See if we can read one less sector.  We will return
					// FERR_IO_END_OF_FILE in this case.

					lSectorCount--;
					rc = RC_SET( FERR_IO_END_OF_FILE);
					goto Try_Read;
				}
				rc = DfsMapError( lResult, FERR_DIRECT_READING_FILE);
				goto Exit;
			}
		}
		uiTotal = (FLMUINT)(lSectorCount * NLM_SECTOR_SIZE);
		pucBuffer += uiTotal;
		m_uiCurrentPos += uiTotal;

		if( puiBytesReadRV)
		{
   		(*puiBytesReadRV) += uiTotal;
		}
		uiBytesToRead -= uiTotal;

		/*
		See if we got everything we wanted to or could with this read.
		*/

		if ((!uiBytesToRead) || (rc == FERR_IO_END_OF_FILE))
			goto Exit;

		/* Go to the next sector after the ones we just read. */

		lStartSector += lSectorCount;
	}

	/*
	At this point, we have less than a sector's worth to read, so we must
	read it into a local buffer.
	*/

	if (m_bNSS)
	{
		if ((lStatus = gv_zDIOReadFunc( m_NssKey, 
								(FLMUINT64)lStartSector, 1,
								(FLMUINT)0, NULL, ucSectorBuf)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_READING_FILE);
			goto Exit;
		}
	}
	else
	{
		lResult = DirectReadFile(
			0,
			m_lFileHandle,
			lStartSector,
			1,
			ucSectorBuf
			);
		if (lResult != 0)
		{
			rc = DfsMapError( lResult, FERR_DIRECT_READING_FILE);
			goto Exit;
		}
	}

	/* Copy the part of the sector that was requested into the buffer. */

	m_uiCurrentPos += uiBytesToRead;

	if( puiBytesReadRV)
	{
  		(*puiBytesReadRV) += uiBytesToRead;
	}

	f_memcpy( pucBuffer, ucSectorBuf, uiBytesToRead);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Might call the direct IO routine in the future
Note:		This function assumes that the pvBuffer that is passed in is
			a multiple of a sector size (512 bytes).
****************************************************************************/
RCODE F_FileHdlImp::SectorRead(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if ( m_bDoDirectIO )
	{
		rc = _DirectIOSectorRead( uiReadOffset, uiBytesToRead, 
			pvBuffer, puiBytesReadRV);
	}
	else
	{
		rc = _Read( uiReadOffset, uiBytesToRead, pvBuffer, puiBytesReadRV);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Calls the direct IO Read routine
Note:		This function assumes that the pvBuffer that is passed in is
			a multiple of a sector size (512 bytes).
****************************************************************************/
RCODE F_FileHdlImp::_DirectIOSectorRead(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE		rc = FERR_OK;
	LONG		lStartSector;
	LONG		lSectorCount;
	LONG		lResult;
	FLMINT	lStatus;

	flmAssert( m_bFileOpened == TRUE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiReadOffset == F_IO_CURRENT_POS)
		uiReadOffset = m_uiCurrentPos;

	if (uiReadOffset % NLM_SECTOR_SIZE != 0)
	{
		rc = _Read( uiReadOffset, uiBytesToRead, pvBuffer, puiBytesReadRV);
		goto Exit;
	}

	/* Calculate the starting sector. */

	lStartSector = uiReadOffset / NLM_SECTOR_SIZE;
	lSectorCount = (LONG)(uiBytesToRead / NLM_SECTOR_SIZE);
	if (uiBytesToRead % NLM_SECTOR_SIZE != 0)
		lSectorCount++;

Try_Read:
	if (m_bNSS)
	{
		if ((lStatus = gv_zDIOReadFunc( m_NssKey,
								(FLMUINT64)lStartSector,
								(FLMUINT)lSectorCount,
								(FLMUINT)0, NULL, pvBuffer)) != zOK)
		{
			if ((lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF) &&
				 (lSectorCount > 1))
			{

				// See if we can read one less sector.  We will return
				// FERR_IO_END_OF_FILE in this case.

				lSectorCount--;
				uiBytesToRead = (FLMUINT)(lSectorCount * NLM_SECTOR_SIZE);
				rc = RC_SET( FERR_IO_END_OF_FILE);
				goto Try_Read;
			}
			uiBytesToRead = 0;
			rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_READING_FILE);
			goto Exit;
		}
	}
	else
	{
		lResult = DirectReadFile( 0, m_lFileHandle,
										lStartSector, lSectorCount,(BYTE *)pvBuffer);
		if (lResult != 0)
		{
			if ((lResult == DFSOperationBeyondEndOfFile) &&
					(lSectorCount > 1))
			{
				// See if we can read one less sector.  We will return
				// FERR_IO_END_OF_FILE in this case.

				lSectorCount--;
				uiBytesToRead = (FLMUINT)(lSectorCount * NLM_SECTOR_SIZE);
				rc = RC_SET( FERR_IO_END_OF_FILE);
				goto Try_Read;
			}
			uiBytesToRead = 0;
			rc = DfsMapError( lResult, FERR_DIRECT_READING_FILE);
			goto Exit;
		}
	}
	m_uiCurrentPos += uiBytesToRead;

Exit:

	if( puiBytesReadRV)
	{
  		*puiBytesReadRV = uiBytesToRead;
	}

	return( rc);
}


/****************************************************************************
Desc:		Sets current position of file.
****************************************************************************/
RCODE F_FileHdlImp::Seek(
	FLMUINT		uiOffset,
	FLMINT		iWhence,
	FLMUINT *	puiNewOffset)
{
	RCODE	rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	switch (iWhence)
	{
		case F_IO_SEEK_CUR:
			m_uiCurrentPos += uiOffset;
			break;
		case F_IO_SEEK_SET:
			m_uiCurrentPos = uiOffset;
			break;
		case F_IO_SEEK_END:
			rc = Size( &m_uiCurrentPos );
			if( rc)
			{
				goto Exit;
			}
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
	}
	*puiNewOffset = m_uiCurrentPos;
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Return the size of the file
****************************************************************************/
RCODE F_FileHdlImp::Size(
	FLMUINT *	puiSize)
{
	LONG		lErr;
	LONG		lSize;
	RCODE		rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (m_bNSS)
	{
		FLMINT	lStatus;
		zInfo_s	Info;

		if ((lStatus = gv_zGetInfoFunc( m_NssKey,
								zGET_STORAGE_USED,
								sizeof( Info), zINFO_VERSION_A,
								&Info)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_GETTING_FILE_INFO);
			goto Exit;
		}
		flmAssert( Info.infoVersion == zINFO_VERSION_A);
		*puiSize = (FLMUINT)Info.std.logicalEOF;
	}
	else
	{
		lErr = GetFileSize( 0, m_lFileHandle, &lSize);
		if ( lErr != 0 )
		{
			rc = MapNWtoFlaimError( lErr, FERR_GETTING_FILE_SIZE);
		}
		*puiSize = (FLMUINT)lSize;
	}

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Truncate the file to the indicated size
WARNING: Direct IO methods are calling this method.  Make sure that all changes
			to this method work in direct IO mode.
****************************************************************************/
RCODE F_FileHdlImp::Truncate(
	FLMUINT	uiSize)
{
	LONG		lErr;
	RCODE		rc = FERR_OK;

	flmAssert( m_bFileOpened == TRUE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (m_bNSS)
	{
		FLMINT	lStatus;
		
		if ((lStatus = gv_zSetEOFFunc( m_NssKey, 0, (FLMUINT64)uiSize,
								zSETSIZE_NON_SPARSE_FILE |
								zSETSIZE_NO_ZERO_FILL |
								zSETSIZE_UNDO_ON_ERR)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_TRUNCATING_FILE);
			goto Exit;
		}
	}
	else
	{
		if ((lErr = SetFileSize(
						0,
						m_lFileHandle,
						uiSize,
						TRUE		// Return freed sectors to the OS
						)) != 0)
		{
			rc = MapNWtoFlaimError( lErr, FERR_TRUNCATING_FILE);
			goto Exit;
		}
	}
	if (m_uiCurrentPos > uiSize)
	{
		m_uiCurrentPos = uiSize;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Write to a file
****************************************************************************/
RCODE F_FileHdlImp::Write(
	FLMUINT				uiWriteOffset,
	FLMUINT				uiBytesToWrite,
	const void *		pvBuffer,
	FLMUINT *			puiBytesWrittenRV)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( m_bDoDirectIO)
	{
		rc = _DirectIOWrite( uiWriteOffset, uiBytesToWrite, 
			pvBuffer,	puiBytesWrittenRV);
	}
	else
	{
		rc = _Write( uiWriteOffset, uiBytesToWrite, 
			pvBuffer,	puiBytesWrittenRV);
	}

Exit:

	return( rc );
}


/****************************************************************************
Desc:		Write to a file
****************************************************************************/
RCODE F_FileHdlImp::_Write(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWrittenRV)
{
	RCODE				rc = FERR_OK;
	LONG				lErr;
	FCBType *		fcb;

	flmAssert( m_bFileOpened == TRUE);

	*puiBytesWrittenRV = 0;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}
	else
	{
		m_uiCurrentPos = uiWriteOffset;
	}
	
	if( m_bNSS)
	{
		FLMINT	lStatus;
		FLMUINT	nBytesWritten;

		if( (lStatus = gv_zWriteFunc( m_NssKey, 0, (FLMUINT64)uiWriteOffset,
			(FLMUINT)uiBytesToWrite, pvBuffer, &nBytesWritten)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_WRITING_FILE);
			goto Exit;
		}
		
		if( nBytesWritten != (FLMUINT)uiBytesToWrite)
		{
			rc = RC_SET( FERR_IO_DISK_FULL);
			goto Exit;
		}
	}
	else
	{
		if( (lErr = MapFileHandleToFCB( m_lFileHandle, &fcb )) != 0)
		{
			rc = MapNWtoFlaimError( lErr, FERR_SETTING_UP_FOR_WRITE);
			goto Exit;
		}

		if( (lErr = WriteFile( fcb->Station, m_lFileHandle, uiWriteOffset,
					uiBytesToWrite, (void *)pvBuffer)) != 0)
		{
			rc = MapNWtoFlaimError( lErr, FERR_WRITING_FILE);
			goto Exit;
		}
	}
	
	*puiBytesWrittenRV = uiBytesToWrite;
	m_uiCurrentPos = uiWriteOffset + uiBytesToWrite;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Calls the direct IO Write routine.
****************************************************************************/
RCODE F_FileHdlImp::_DirectIOWrite(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWrittenRV)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucBuffer = (FLMBYTE *)pvBuffer;
	LONG				lStartSector;
	LONG				lSectorCount;
	LONG				lResult;
	BYTE				ucSectorBuf[ NLM_SECTOR_SIZE];
	FLMUINT			uiBytesToCopy;
	FLMUINT			uiSectorOffset;
	FLMUINT			uiTotal;
	FLMINT			lStatus;

	flmAssert( m_bFileOpened == TRUE);

	*puiBytesWrittenRV = 0;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}
	else
	{
		m_uiCurrentPos = uiWriteOffset;
	}
	
	// Calculate the starting sector

	lStartSector = uiWriteOffset / NLM_SECTOR_SIZE;
	

	// See if the offset is on a sector boundary.  If not, we must first read
	// the sector into memory, overwrite it with data from the input
	// buffer and write it back out again.

	if( (uiWriteOffset % NLM_SECTOR_SIZE != 0) || 
		 (uiBytesToWrite < NLM_SECTOR_SIZE))
	{
		if( m_bNSS)
		{
			if( (lStatus = gv_zDIOReadFunc( m_NssKey, 
				(FLMUINT64)lStartSector,
				(FLMUINT)1, (FLMUINT)0, NULL, ucSectorBuf)) != zOK)
			{
				if (lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF)
				{
					f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));

					// Expand the file

					if( RC_BAD( rc = Expand( lStartSector, 1)))
					{
						goto Exit;
					}
				}
				else
				{
					rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_READING_FILE);
					goto Exit;
				}
			}
		}
		else
		{
			lResult = DirectReadFile( 0, m_lFileHandle, lStartSector, 
												1, ucSectorBuf);
												
			if( lResult == DFSHoleInFileError || 
				 lResult == DFSOperationBeyondEndOfFile )
			{
				f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));

				// Expand the file

				if( RC_BAD( rc = Expand( lStartSector, 1)))
				{
					goto Exit;
				}
			}
			else if( lResult != 0)
			{
				rc = DfsMapError( lResult, FERR_DIRECT_READING_FILE);
				goto Exit;
			}
		}

		// Copy the part of the buffer that is being written back into
		// the sector buffer.

		uiSectorOffset = uiWriteOffset % NLM_SECTOR_SIZE;

		if( (uiBytesToCopy = uiBytesToWrite) > NLM_SECTOR_SIZE - uiSectorOffset)
		{
			uiBytesToCopy = NLM_SECTOR_SIZE - uiSectorOffset;
		}
		
		f_memcpy( &ucSectorBuf [uiSectorOffset], pucBuffer, uiBytesToCopy);
		pucBuffer += uiBytesToCopy;
		uiBytesToWrite -= (FLMUINT)uiBytesToCopy;
		m_uiCurrentPos += (FLMUINT)uiBytesToCopy;
		(*puiBytesWrittenRV) += (FLMUINT)uiBytesToCopy;

		// Write the sector buffer back out

		if( RC_BAD( rc = WriteSectors( &ucSectorBuf [0], lStartSector, 1, NULL)))
		{
			goto Exit;
		}

		// See if we wrote everything we wanted to with this write

		if (!uiBytesToWrite)
		{
			goto Exit;
		}

		// Go to the next sector boundary

		lStartSector++;
	}

	// At this point, we are poised to write on a sector boundary.  See if we
	// have at least one full sector to write.  If so, we can write it directly
	// from the provided buffer.  If not, we must use the temporary sector
	// buffer.

	if( uiBytesToWrite >= NLM_SECTOR_SIZE)
	{
		lSectorCount = (LONG)(uiBytesToWrite / NLM_SECTOR_SIZE);
		
		if( RC_BAD( rc = WriteSectors( (void *)pucBuffer, lStartSector,
									lSectorCount, NULL)))
		{
			goto Exit;
		}
		
		uiTotal = (FLMUINT)(lSectorCount * NLM_SECTOR_SIZE);
		pucBuffer += uiTotal;
		m_uiCurrentPos += uiTotal;
		(*puiBytesWrittenRV) += uiTotal;
		uiBytesToWrite -= uiTotal;

		// See if we wrote everything we wanted to with this write

		if( !uiBytesToWrite)
		{
			goto Exit;
		}

		// Go to the next sector after the ones we just wrote

		lStartSector += lSectorCount;
	}

	// At this point, we have less than a sector's worth to write, so we must
	// first read the sector from disk, alter it, and then write it back out.

	if( m_bNSS)
	{
		if( (lStatus = gv_zDIOReadFunc( m_NssKey, (FLMUINT64)lStartSector,
			(FLMUINT)1, (FLMUINT)0, NULL, ucSectorBuf)) != zOK)
		{
			if( lStatus == zERR_END_OF_FILE || lStatus == zERR_BEYOND_EOF)
			{
				f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));

				// Expand the file

				if( RC_BAD( rc = Expand( lStartSector, 1)))
				{
					goto Exit;
				}
			}
			else
			{
				rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_READING_FILE);
				goto Exit;
			}
		}
	}
	else
	{
		lResult = DirectReadFile( 0, m_lFileHandle, lStartSector,
											1, ucSectorBuf);
											
		if( lResult == DFSHoleInFileError)
		{
			f_memset( ucSectorBuf, 0, sizeof( ucSectorBuf));

			// Expand the file

			if( RC_BAD( rc = Expand( lStartSector, 1)))
			{
				goto Exit;
			}
		}
		else if( lResult != 0)
		{
			rc = DfsMapError( lResult, FERR_DIRECT_READING_FILE);
			goto Exit;
		}
	}

	// Copy the rest of the output buffer into the sector buffer

	f_memcpy( ucSectorBuf, pucBuffer, uiBytesToWrite);

	// Write the sector back to disk

	if( RC_BAD( rc = WriteSectors( &ucSectorBuf [0], lStartSector, 1, NULL)))
	{
		goto Exit;
	}

	m_uiCurrentPos += uiBytesToWrite;
	(*puiBytesWrittenRV) += uiBytesToWrite;
	
Exit:

	return( rc);
}

/***************************************************************************
Desc:		Expand a file for writing.
***************************************************************************/
RCODE F_FileHdlImp::Expand(
	LONG		lStartSector,
	LONG		lSectorsToAlloc)
{
	RCODE			rc = FERR_OK;
	LONG			lResult;
	LONG			lBlockNumber;
	LONG			lStartBlockNumber;
	LONG			lNumBlocksToAlloc;
	LONG			lNumBlocksAllocated;
	LONG			lMinToAlloc;
	LONG			lLastBlockNumber;
	LONG			lTotalToAlloc;
	LONG			lExtendSize;
	FLMUINT		uiFileSize;
	FLMUINT		uiRequestedExtendSize = m_uiExtendSize;
	FLMBOOL		bVerifyFileSize = FALSE;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	// If the requested extend size is the "special" value of ~0,
	// we will set the requested size to 0, so that we will use the
	// minimum default below.  This allows us to somewhat emulate what
	// the Window's code does.

	if( uiRequestedExtendSize == (FLMUINT)(~0))
	{
		uiRequestedExtendSize = 0;
	}

	if( m_bNSS)
	{
		lStartBlockNumber = lStartSector / (65536 / NLM_SECTOR_SIZE);
		lLastBlockNumber = (lStartSector + lSectorsToAlloc) / (65536 / NLM_SECTOR_SIZE);
		lExtendSize = uiRequestedExtendSize / 65536;
	}
	else
	{
		lStartBlockNumber = lStartSector / m_lSectorsPerBlock;
		lLastBlockNumber = (lStartSector + lSectorsToAlloc) / m_lSectorsPerBlock;
		lExtendSize = uiRequestedExtendSize / (m_lSectorsPerBlock * NLM_SECTOR_SIZE);
	}

	// Last block number better be greater than or equal to
	// start block number.

	flmAssert( lLastBlockNumber >= lStartBlockNumber);
	lMinToAlloc = (lLastBlockNumber - lStartBlockNumber) + 1;

	if( lExtendSize < 5)
	{
		lExtendSize = 5;
	}

	// Allocate up to lExtendSize blocks at a time - hopefully this will be
	// more efficient.

	if( lMinToAlloc < lExtendSize)
	{
		lTotalToAlloc = lExtendSize;
	}
	else if( lMinToAlloc % lExtendSize == 0)
	{
		lTotalToAlloc = lMinToAlloc;
	}
	else
	{
		// Keep the total blocks to allocate a multiple of lExtendSize.

		lTotalToAlloc = lMinToAlloc - 
			(lMinToAlloc % lExtendSize) + lExtendSize;
	}
	
	lNumBlocksToAlloc = lTotalToAlloc;
	lBlockNumber = lStartBlockNumber;
	lNumBlocksAllocated = 0;

	// Must not go over maximum file size.

	if( lStartBlockNumber + lTotalToAlloc > m_lMaxBlocks)
	{
		lNumBlocksToAlloc = lTotalToAlloc = m_lMaxBlocks - lStartBlockNumber;
		
		if( lTotalToAlloc < lMinToAlloc)
		{
			rc = RC_SET( FERR_IO_DISK_FULL);
			goto Exit;
		}
	}

	if( m_bNSS)
	{
		FLMINT	lStatus;

		for( ;;)
		{
			if( (lStatus = gv_zSetEOFFunc( m_NssKey, 0,
				(FLMUINT64)lBlockNumber * 65536 + lNumBlocksToAlloc * 65536,
				zSETSIZE_NO_ZERO_FILL | zSETSIZE_NON_SPARSE_FILE)) != zOK)
			{
				if( lStatus == zERR_OUT_OF_SPACE)
				{
					if( lNumBlocksToAlloc > lMinToAlloc)
					{
						lNumBlocksToAlloc--;
						continue;
					}
				}
				
				rc = MapNSSToFlaimError( lStatus, FERR_EXPANDING_FILE);
				goto Exit;
			}
			else
			{
				break;
			}
		}
	}
	else
	{
		for (;;)
		{
			lResult = ExpandFileInContiguousBlocks( 0, m_lFileHandle, 
									lBlockNumber, lNumBlocksToAlloc, -1, -1);

			// If we couldn't allocate space, see if we can free some of
			// the limbo space on the volume.

			if( lResult == DFSInsufficientSpace || lResult == DFSBoundryError)
			{
				// May not have been able to get contiguous space for
				// multiple blocks.  If we were asking for more than
				// one, reduce the number we are asking for and try
				// again.

				if( lNumBlocksToAlloc > 1)
				{
					lNumBlocksToAlloc--;
					continue;
				}

				// If we could not even get one block, it is time to
				// try and free some limbo space.

				lResult = FreeLimboVolumeSpace( (LONG)m_lVolumeID, 1);
				if( lResult == DFSInsufficientLimboFileSpace)
				{
					// It is not an error to be out of space if
					// we successfully allocated at least the minimum
					// number of blocks needed.

					if( lNumBlocksAllocated >= lMinToAlloc)
					{
						break;
					}
					else
					{
						rc = RC_SET( FERR_IO_DISK_FULL);
						goto Exit;
					}
				}
				
				continue;
			}
			else if( lResult == DFSOverlapError)
			{
				lResult = 0;
				bVerifyFileSize = TRUE;

				// If lNumBlocksToAlloc is greater than one, we
				// don't know exactly where the hole is, so we need
				// to try filling exactly one block right where
				// we are at.

				// If lNumBlocksToAlloc is exactly one, we know that
				// we have a block right where we are at, so we let
				// the code fall through as if the expand had
				// succeeded.

				if( lNumBlocksToAlloc > 1)
				{
					// If we have an overlap, try getting one block at
					// the current block number - need to make sure this
					// is not where the hole is at.

					lNumBlocksToAlloc = 1;
					continue;
				}
			}
			else if (lResult != 0)
			{
				rc = DfsMapError( lResult, FERR_EXPANDING_FILE);
				goto Exit;
			}
			
			lNumBlocksAllocated += lNumBlocksToAlloc;
			lBlockNumber += lNumBlocksToAlloc;
			
			if( lNumBlocksAllocated >= lTotalToAlloc)
			{
				break;
			}
			else if( lNumBlocksToAlloc > lTotalToAlloc - lNumBlocksAllocated)
			{
				lNumBlocksToAlloc = lTotalToAlloc - lNumBlocksAllocated;
			}
		}

		// If bVerifyFileSize is TRUE, we had an overlap error, which means
		// that we may have had a hole in the file.  In that case, we
		// do NOT want to truncate the file to an incorrect size, so we
		// get the current file size to make sure we are not reducing it
		// down inappropriately.  NOTE: This is not foolproof - if we have
		// a hole that is exactly the size we asked for, we will not verify
		// the file size.

		uiFileSize = (FLMUINT)(lStartBlockNumber + lNumBlocksAllocated) *
				(FLMUINT)m_lSectorsPerBlock * (FLMUINT)NLM_SECTOR_SIZE;
				
		if( bVerifyFileSize)
		{
			LONG	lCurrFileSize;

			lResult = GetFileSize( 0, m_lFileHandle, &lCurrFileSize);
			
			if( lResult != DFSNormalCompletion)
			{
				rc = DfsMapError( lResult, FERR_GETTING_FILE_SIZE);
				goto Exit;
			}
			
			if( (FLMUINT)lCurrFileSize > uiFileSize)
			{
				uiFileSize = (FLMUINT)lCurrFileSize;
			}
		}

		// This call of SetFileSize is done to force the directory entry file size
		// to account for the newly allocated blocks.  It also forces the directory
		// entry to be updated on disk.  If we didn't do this here, the directory
		// entry's file size on disk would not account for this block.
		// Thus, if we crashed after writing data to this
		// newly allocated block, we would lose the data in the block.

		lResult = SetFileSize( 0, m_lFileHandle, uiFileSize, FALSE);
		
		if( lResult != DFSNormalCompletion)
		{
			rc = DfsMapError( lResult, FERR_TRUNCATING_FILE);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Calls the direct IO Write routine.  Handles both asynchronous writes
			and synchronous writes.
****************************************************************************/
RCODE F_FileHdlImp::WriteSectors(
	void *			pvBuffer,
	LONG				lStartSector,
	LONG				lSectorCount,
	F_IOBuffer *	pBufferObj,
	FLMBOOL *		pbDidAsync)
{
	RCODE				rc = FERR_OK;
	LONG				lResult;
	FLMBOOL			bAlreadyExpanded = FALSE;
	FLMINT			lStatus;
	FLMBOOL			bMadePending;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	// Keep trying write until we succeed or get an error we can't deal with.
	// Actually, this will NOT loop forever.  At most, it will try twice - 
	// and then it is only when we get a hole in the file error.

	bMadePending = FALSE;
	for (;;)
	{
		if (m_bNSS)
		{
			if( pBufferObj)
			{
				if (!bMadePending)
				{
					flmAssert( pbDidAsync);
					pBufferObj->makePending();
					bMadePending = TRUE;
				}
				lStatus = gv_zDIOWriteFunc( m_NssKey,
									(FLMUINT64)lStartSector,
									(FLMUINT)lSectorCount,
									(FLMUINT)pBufferObj,
									nssDioCallback,
									pvBuffer);
			}
			else
			{
				lStatus = gv_zDIOWriteFunc( m_NssKey,
									(FLMUINT64)lStartSector,
									(FLMUINT)lSectorCount, (FLMUINT)0, NULL, pvBuffer);
			}

			// We may need to allocate space to do this write

			if (lStatus == zERR_END_OF_FILE ||
				 lStatus == zERR_BEYOND_EOF || 
				 lStatus == zERR_HOLE_IN_DIO_FILE)
			{
				if (bAlreadyExpanded)
				{
					flmAssert( 0);
					rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_WRITING_FILE);
					goto Exit;
				}

				// Expand the file

				if (RC_BAD( rc = Expand( lStartSector, lSectorCount)))
				{
					goto Exit;
				}
				bAlreadyExpanded = TRUE;
				continue;
			}
			else if (lStatus != 0)
			{
				rc = MapNSSToFlaimError( lStatus, FERR_DIRECT_WRITING_FILE);
				goto Exit;
			}
			else
			{
				if (pBufferObj)
				{
					*pbDidAsync = TRUE;
				}
				break;
			}
		}
		else
		{
			LONG		lSize;
			FLMBOOL	bNeedToWriteEOF;

			// Determine if this write will change the EOF.  If so, pre-expand
			// the file.

			lResult = GetFileSize( 0, m_lFileHandle, &lSize);
			if (lResult != 0)
			{
				rc = MapNWtoFlaimError( lResult, FERR_GETTING_FILE_SIZE);
				goto Exit;
			}
			
			bNeedToWriteEOF = (lSize < (lStartSector + lSectorCount) * NLM_SECTOR_SIZE)
											? TRUE
											: FALSE;
											
			if( pBufferObj)
			{
				if (!bMadePending)
				{
					flmAssert( pbDidAsync);
					pBufferObj->makePending();
					bMadePending = TRUE;
				}
				
				lResult = DirectWriteFileNoWait( 0, m_lFileHandle,
									lStartSector,lSectorCount,
									(BYTE *)pvBuffer, DirectIONoWaitCallBack, 
									(LONG)pBufferObj);
			}
			else
			{
				lResult = DirectWriteFile( 0, m_lFileHandle,
									lStartSector, lSectorCount, (BYTE *)pvBuffer);
			}

			// We may need to allocate space to do this write

			if( lResult == DFSHoleInFileError || 
				 lResult == DFSOperationBeyondEndOfFile)
			{
				if( bAlreadyExpanded)
				{
					flmAssert( 0);
					rc = DfsMapError( lResult, FERR_DIRECT_WRITING_FILE);
					goto Exit;
				}

				// Expand the file

				if( RC_BAD( rc = Expand( lStartSector, lSectorCount)))
				{
					goto Exit;
				}
				
				bAlreadyExpanded = TRUE;

				// The Expand method forces the file EOF in the directory
				// entry to be written to disk.

				bNeedToWriteEOF = FALSE;
				continue;
			}
			else if (lResult != 0)
			{
				rc = DfsMapError( lResult, FERR_DIRECT_WRITING_FILE);
				goto Exit;
			}
			else
			{
				if( pBufferObj)
				{
					*pbDidAsync = TRUE;
				}

				// If bNeedToWriteEOF is TRUE, we need to force EOF to disk.

				if( bNeedToWriteEOF)
				{
					LONG	lFileSizeInSectors;
					LONG	lExtraSectors;

					// Set the EOF to the nearest block boundary - so we don't
					// have to do this very often.

					lFileSizeInSectors = lStartSector + lSectorCount;
					lExtraSectors = lFileSizeInSectors % m_lSectorsPerBlock;
					
					if (lExtraSectors)
					{
						lFileSizeInSectors += (m_lSectorsPerBlock - lExtraSectors);
					}
					
					if ((lResult = SetFileSize( 0, m_lFileHandle,
											(FLMUINT)lFileSizeInSectors * NLM_SECTOR_SIZE,
											FALSE)) != 0)
					{
						rc = DfsMapError( lResult, FERR_TRUNCATING_FILE);
						goto Exit;
					}
				}

				break;
			}
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Legacy async I/O completion callback
****************************************************************************/
FSTATIC void DirectIONoWaitCallBack(
	LONG		unknownAlwaysZero,
	LONG		callbackData,
	LONG 		completionCode)
{
	F_IOBuffer *	pIOBuffer = (F_IOBuffer *)callbackData;

	F_UNREFERENCED_PARM( unknownAlwaysZero);

	pIOBuffer->signalComplete(
		(RCODE)(completionCode == DFSNormalCompletion
				  ? FERR_OK
				  : DfsMapError( completionCode, FERR_DIRECT_WRITING_FILE)));
}

/****************************************************************************
Desc: NSS async I/O completion callback
****************************************************************************/
FSTATIC void nssDioCallback(
	FLMUINT	reserved,
	FLMUINT	callbackData,
	FLMUINT	completionCode)
{
	F_IOBuffer *	pIOBuffer = (F_IOBuffer *)callbackData;

	F_UNREFERENCED_PARM( reserved);

	pIOBuffer->signalComplete( 
		(RCODE)(completionCode == zOK
				  ? FERR_OK
				  : MapNSSToFlaimError( completionCode, FERR_DIRECT_WRITING_FILE)));
}

/****************************************************************************
Desc:		Might call the direct IO Write routine in the future
Note:		This routine assumes that the size of pvBuffer is a multiple of
			sector size (512 bytes) and can be used to write out full
			sectors.  Even if uiBytesToWrite does not account for full sectors,
			data from the buffer will still be written out - a partial sector
			on disk will not be preserved.
****************************************************************************/
RCODE F_FileHdlImp::SectorWrite(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT			uiBufferSize,
	F_IOBuffer *	pBufferObj,
	FLMUINT *		puiBytesWrittenRV,
	FLMBOOL			bZeroFill)
{
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( uiBufferSize);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if ( m_bDoDirectIO)
	{
		rc = _DirectIOSectorWrite( uiWriteOffset, uiBytesToWrite, pvBuffer,
							pBufferObj, puiBytesWrittenRV, bZeroFill);
	}
	else
	{
		flmAssert( pBufferObj == NULL);
		rc = _Write( uiWriteOffset, uiBytesToWrite, pvBuffer,	puiBytesWrittenRV);
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Calls the direct IO-style write routine.
Note:		This routine assumes that the size of pvBuffer is a multiple of
			sector size (512 bytes) and can be used to write out full
			sectors.  Even if uiBytesToWrite does not account for full sectors,
			data from the buffer will still be written out - a partial sector
			on disk will not be preserved.
****************************************************************************/
RCODE F_FileHdlImp::_DirectIOSectorWrite(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	F_IOBuffer *	pBufferObj,
	FLMUINT *		puiBytesWrittenRV,
	FLMBOOL			bZeroFill)
{
	RCODE		rc = FERR_OK;
	LONG		lStartSector;
	LONG		lSectorCount;
	FLMBOOL	bDidAsync = FALSE;

	flmAssert( m_bFileOpened == TRUE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}
	else
	{
		m_uiCurrentPos = uiWriteOffset;
	}
	
	if (uiWriteOffset % NLM_SECTOR_SIZE != 0)
	{
		rc = Write( uiWriteOffset, uiBytesToWrite, pvBuffer,
							puiBytesWrittenRV);
		goto Exit;
	}

	// Calculate the starting sector and number of sectors to write

	lStartSector = uiWriteOffset / NLM_SECTOR_SIZE;
	lSectorCount = (LONG)(uiBytesToWrite / NLM_SECTOR_SIZE);
	if (uiBytesToWrite % NLM_SECTOR_SIZE != 0)
	{
		FLMBYTE *	pucBuffer = (FLMBYTE *)pvBuffer;

		lSectorCount++;

		if (bZeroFill)
		{
			// Zero out the part of the buffer that was not included in
			// uiBytesToWrite - because it will still be written to disk.

			f_memset( &pucBuffer [uiBytesToWrite], 0,
						(FLMUINT)(NLM_SECTOR_SIZE - (uiBytesToWrite % NLM_SECTOR_SIZE)));
		}
	}

	if( RC_BAD( rc = WriteSectors( (void *)pvBuffer, lStartSector, lSectorCount,
											pBufferObj, &bDidAsync)))
	{
		goto Exit;
	}

	m_uiCurrentPos += uiBytesToWrite;

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = uiBytesToWrite;
	}

Exit:

	if( !bDidAsync && pBufferObj)
	{
		pBufferObj->notifyComplete( rc);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Determine if a file or directory exists
****************************************************************************/
RCODE NWTestIfFileExists(
	const char *	pPath)
{
	RCODE			rc = FERR_OK;
	LONG			unused;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	LONG			lErrorCode;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	f_strcpy( (char *)&ucPseudoLNamePath[1], pPath);
	ucPseudoLNamePath[0] = (char)f_strlen( pPath);
	
	if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID,		
		&lPathID, ucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}

	if( (lErrorCode = MapPathToDirectoryNumber( 0, lVolumeID, 0, ucLNamePath,
		lLNamePathCount, LONGNameSpace, &lDirectoryID, &unused)) != 0)
	{
		goto Exit;
	}

Exit:

	if( lErrorCode == 255 || lErrorCode == 156)
	{
		// Too many error codes map to 255, so we put in a special
		// case check here

		rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
	}
	else if( lErrorCode )
	{
		rc = MapNWtoFlaimError( lErrorCode, FERR_CHECKING_FILE_EXISTENCE);
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Delete a file
****************************************************************************/
RCODE NWDeleteFile(
	const char *	pPath)
{
	RCODE			rc = FERR_OK;
	LONG			lErrorCode;
	char			pszQualifiedPath[ F_PATH_MAX_SIZE];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG			lLNamePathCount;
	LONG			lVolumeID;
	FLMBOOL		bNssVolume = FALSE;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	ConvertToQualifiedNWPath( pPath, pszQualifiedPath);

	if( (lErrorCode = ConvertPathToLNameFormat( pszQualifiedPath, &lVolumeID,
			&bNssVolume, ucLNamePath, &lLNamePathCount)) != 0)
	{
		rc = MapNWtoFlaimError( lErrorCode, FERR_RENAMING_FILE);
		goto Exit;
	}

	if( gv_bNSSKeyInitialized && bNssVolume)
	{
		if( (lErrorCode = gv_zDeleteFunc( gv_NssRootKey, 0,
								zNSPACE_LONG | zMODE_UTF8,
								pszQualifiedPath, zMATCH_ALL, 0)) != zOK)
		{
			rc = MapNSSToFlaimError( lErrorCode, FERR_DELETING_FILE);
			goto Exit;
		}
	}
	else
	{
		if( (lErrorCode = EraseFile( 0, 1, lVolumeID, 0, ucLNamePath,
			lLNamePathCount, LONGNameSpace, 0)) != 0)
		{
			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
			}
			else
			{
				rc = MapNWtoFlaimError( lErrorCode, FERR_DELETING_FILE);
			}
			
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Turn off the rename inhibit bit for a file in an NSS volume.
****************************************************************************/
FSTATIC RCODE nssTurnOffRenameInhibit(
	const char *	pszFileName)
{
	RCODE				rc = FERR_OK;
	zInfo_s			Info;
	FLMINT64			NssKey;
	FLMBOOL			bFileOpened = FALSE;
	FLMINT			lStatus;
	FLMUINT			nOpenAttr;

	nOpenAttr = FlaimToNSSOpenFlags( (FLMUINT)(F_IO_RDWR |
															F_IO_SH_DENYNONE), FALSE);
															
	if( (lStatus = gv_zOpenFunc( gv_NssRootKey, 1, zNSPACE_LONG | zMODE_UTF8,
					pszFileName, nOpenAttr, &NssKey)) != zOK)
	{
		rc = MapNSSToFlaimError( lStatus, FERR_OPENING_FILE);
		goto Exit;
	}
	
	bFileOpened = TRUE;

	// Get the file attributes.

	if( (lStatus = gv_zGetInfoFunc( NssKey, zGET_STD_INFO, sizeof( Info),
				zINFO_VERSION_A, &Info)) != zOK)
	{
		rc = MapNSSToFlaimError( lStatus, FERR_GETTING_FILE_INFO);
		goto Exit;
	}
	
	flmAssert( Info.infoVersion == zINFO_VERSION_A);

	// See if the rename inhibit bit is set.

	if( Info.std.fileAttributes & zFA_RENAME_INHIBIT)
	{
		// Turn bit off

		Info.std.fileAttributes = 0;

		// Specify which bits to modify - only rename inhibit in this case

		Info.std.fileAttributesModMask = zFA_RENAME_INHIBIT;

		if( (lStatus = gv_zModifyInfoFunc( NssKey, 0, zMOD_FILE_ATTRIBUTES,
			sizeof( Info), zINFO_VERSION_A, &Info)) != zOK)
		{
			rc = MapNSSToFlaimError( lStatus, FERR_SETTING_FILE_INFO);
			goto Exit;
		}
	}
	
Exit:

	if( bFileOpened)
	{
		(void)gv_zCloseFunc( NssKey);
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Rename a file
Notes:	Currently, this function doesn't support moving the file from one
			volume to another.  (There is a CopyFileToFile function that could
			be used to do the move.)  The toolkit function does appear to 
			support moving (copy/delete) the file.
			
			This function does support renaming directories.
****************************************************************************/
RCODE NWRenameFile(
	const char *	pOldFilePath,
	const char *	pNewFilePath)
{
	RCODE			rc = FERR_OK;
	LONG			unused;
	FLMBYTE		ucOldLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG			lOldLNamePathCount;
	FLMBYTE		ucNewLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG			lNewLNamePathCount;
	LONG			lVolumeID;
	LONG			lErrorCode;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG			lPathID;
	LONG			lIsFile;
	FLMBOOL		bIsDirectory;
	struct 		ModifyStructure modifyStruct;
	LONG			lDirectoryID;
	LONG			lFileAttributes;
	LONG			lMatchBits;
	FLMBOOL		bNssVolume =
						(FLMBOOL)(gv_zIsNSSVolumeFunc
							? (gv_zIsNSSVolumeFunc( (const char *)pOldFilePath)
									 ? TRUE
									 : FALSE)
							: FALSE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( gv_bNSSKeyInitialized && bNssVolume)
	{
		FLMINT	lStatus;
		FLMBOOL	bTurnedOffRenameInhibit = FALSE;

Retry_Nss_Rename:

		if( (lStatus = gv_zRenameFunc( gv_NssRootKey, 0,
			zNSPACE_LONG | zMODE_UTF8, pOldFilePath, zMATCH_ALL,
			zNSPACE_LONG | zMODE_UTF8, pNewFilePath, 0)) != zOK)
		{
			if( lStatus == zERR_NO_RENAME_PRIVILEGE && !bTurnedOffRenameInhibit)
			{
				// Attempt to turn off rename inhibit.  This isn't always the
				// reason for zERR_NO_RENAME_PRIVILEGE, but it is one we
				// definitely need to take care of.

				if( RC_BAD( rc = nssTurnOffRenameInhibit( pOldFilePath)))
				{
					goto Exit;
				}
				
				bTurnedOffRenameInhibit = TRUE;
				goto Retry_Nss_Rename;
			}
			
			rc = MapNSSToFlaimError( lStatus, FERR_RENAMING_FILE);
			goto Exit;
		}
	}
	else
	{
		f_strcpy( (char *)&ucPseudoLNamePath[1], pOldFilePath);
		ucPseudoLNamePath[0] = (char)f_strlen( (const char *)&ucPseudoLNamePath[1] );
		
		if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID,		
			&lPathID, (BYTE *)ucOldLNamePath, &lOldLNamePathCount)) != 0)
		{
			goto Exit;
		}

		if( (lErrorCode = MapPathToDirectoryNumber( 0, lVolumeID, 0,
			(BYTE *)ucOldLNamePath, lOldLNamePathCount, LONGNameSpace,
			&lDirectoryID, &lIsFile)) != 0)
		{
			goto Exit;
		}
		
		if( lIsFile)
		{
			bIsDirectory = FALSE;
			lMatchBits = 0;
		}
		else
		{
			bIsDirectory = TRUE;
			lMatchBits = SUBDIRECTORY_BIT;
		}
		
		f_strcpy( (char *)&ucPseudoLNamePath[1], pNewFilePath);
		ucPseudoLNamePath[0] = (char)f_strlen( (const char *)&ucPseudoLNamePath[1]);
		
		if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &unused,
			&lPathID, (BYTE *)ucNewLNamePath, &lNewLNamePathCount)) != 0)
		{
			goto Exit;
		}

		{
			struct DirectoryStructure * pFileInfo;

			if( (lErrorCode = VMGetDirectoryEntry( lVolumeID, 
				lDirectoryID & 0x00ffffff, &pFileInfo)) != 0)
			{
				goto Exit;
			}
			
			lFileAttributes = pFileInfo->DFileAttributes;
		}
		
		if( lFileAttributes & RENAME_INHIBIT_BIT )
		{
			f_memset(&modifyStruct, 0, sizeof(modifyStruct));
			modifyStruct.MFileAttributesMask = RENAME_INHIBIT_BIT;
			
			if( (lErrorCode = ModifyDirectoryEntry( 0, 1, lVolumeID, 0,
				(BYTE *)ucOldLNamePath, lOldLNamePathCount, LONGNameSpace, 
				lMatchBits, LONGNameSpace, &modifyStruct,
				MFileAttributesBit, 0)) != 0)
			{
				goto Exit;
			}
		}

		lErrorCode = RenameEntry( 0, 1, lVolumeID, 0, ucOldLNamePath,
			lOldLNamePathCount, LONGNameSpace, lMatchBits,
			(BYTE)bIsDirectory ? 1 : 0, 0, ucNewLNamePath, lNewLNamePathCount,
			TRUE, TRUE);

		if( lFileAttributes & RENAME_INHIBIT_BIT )
		{
			FLMBYTE *		pFileName;

			if( lErrorCode )
			{
				pFileName = ucOldLNamePath;
				lNewLNamePathCount = lOldLNamePathCount;
			}
			else
			{
				pFileName = ucNewLNamePath;
			}
				
			// Turn the RENAME_INHIBIT_BIT back on
			
			f_memset(&modifyStruct, 0, sizeof(modifyStruct));
			modifyStruct.MFileAttributes = RENAME_INHIBIT_BIT;
			modifyStruct.MFileAttributesMask = RENAME_INHIBIT_BIT;

			(void)ModifyDirectoryEntry( 0, 1, lVolumeID, 0, (BYTE *)pFileName,
				lNewLNamePathCount, LONGNameSpace, lMatchBits, LONGNameSpace,
				&modifyStruct, MFileAttributesBit, 0);
		}
	}

Exit:

	if( !gv_bNSSKeyInitialized || !bNssVolume)
	{
		if( lErrorCode )
		{
			// Too many error codes map to 255, so we put in a special
			// case check here.

			if( lErrorCode == 255)
			{
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
			}
			else
			{
				rc = MapNWtoFlaimError( lErrorCode, FERR_RENAMING_FILE);
			}
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Convert the given path to NetWare LName format.
Input:	pPath = qualified netware path of the format:
						volume:directory_1\...\directory_n\filename.ext
Output:	plVolumeID = NetWare volume ID
			pLNamePath = NetWare LName format path
											 
				Netware expects paths to be in LName format:
					<L1><C1><L2><C2>...<Ln><Cn>
					where <Lx> is a one-byte length and <Cx> is a path component.
					
					Example: 6SYSTEM4Fred
						note that the 6 and 4 are binary, not ASCII

			plLNamePathCount = number of path components in pLNamePath
****************************************************************************/
FSTATIC LONG ConvertPathToLNameFormat(
	const char *		pPath,
	LONG *				plVolumeID,
	FLMBOOL *			pbNssVolume,
	FLMBYTE *			pLNamePath,
	LONG *				plLNamePathCount)
{
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	LONG			lPathID;
	LONG			lErrorCode = 0;

	*pLNamePath = 0;
	*plLNamePathCount = 0;

#ifdef FLM_DEBUG
	if( RC_BAD( GET_FS_ERROR()))
	{
		lErrorCode = 255;
		goto Exit;
	}
#endif

	*pbNssVolume = (FLMBOOL)(gv_zIsNSSVolumeFunc
									? (gv_zIsNSSVolumeFunc( (const char *)pPath)
										? TRUE
										: FALSE)
									: FALSE);

	if( gv_bNSSKeyInitialized && *pbNssVolume)
	{
		f_strcpy( (char *)pLNamePath, pPath);
		*plLNamePathCount = 1;
	}
	else
	{
		f_strcpy( (char *)&ucPseudoLNamePath[1], pPath);
		ucPseudoLNamePath[0] = (FLMBYTE)f_strlen( (const char *)&ucPseudoLNamePath[1]);
		
		if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, plVolumeID,		
			&lPathID, (BYTE *)pLNamePath, plLNamePathCount)) != 0)
		{
			goto Exit;
		}
	}

Exit:

	return( lErrorCode );
}

/****************************************************************************
Desc:		Convert the given path to a NetWare format.  The format isn't 
			critical, it just needs to be consistent.  See below for a 
			description of the format chosen.
Input:	pInputPath = a path to a file
Output:	pszQualifiedPath = qualified netware path of the format:
									volume:directory_1\...\directory_n\filename.ext
									
			If no volume is given, "SYS:" is the default.
****************************************************************************/
FSTATIC void ConvertToQualifiedNWPath(
	const char *	pInputPath,
	char *			pszQualifiedPath)
{
	char		ucFileName [F_FILENAME_SIZE];
	char		ucVolume [MAX_NETWARE_VOLUME_NAME+1];
	char		ucPath [F_PATH_MAX_SIZE + 1];

	// Separate path into its components: volume, path...

	f_pathParse( pInputPath, NULL, ucVolume, ucPath, ucFileName);

	// Rebuild path to a standard, fully-qualified format, defaulting the
	// volume if one isn't specified.

	*pszQualifiedPath = 0;
	if( ucVolume [0])
	{
		// Append the volume specified by the user.

		f_strcat( pszQualifiedPath, ucVolume );
	}
	else
	{
		// No volume specified, use the default

		f_strcat( pszQualifiedPath, "SYS:");
	}
	
	if( ucPath [0])
	{
		// User specified a path...

		if( ucPath[0] == '\\' || ucPath[0] == '/' )
		{
			// Append the path to the volume without the leading slash

			f_strcat( pszQualifiedPath, &ucPath [1]);
		}
		else
		{
			// Append the path to the volume

			f_strcat( pszQualifiedPath, ucPath);
		}
	}

	if( ucFileName [0])
	{
		// Append the file name to the path

		f_pathAppend( pszQualifiedPath, ucFileName);
	}
}

#endif
