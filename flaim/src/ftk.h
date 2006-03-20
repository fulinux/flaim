//-------------------------------------------------------------------------
// Desc:	Toolkit - cross platform APIs for system functionality.
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
// $Id: ftk.h 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef FTK_H
#define FTK_H

#ifdef FLM_NLM
	
	#ifndef LONG
		#define LONG	unsigned long
	#endif
	
	#ifndef WORD
		#define WORD	unsigned short
	#endif
	
	#ifndef BYTE
		#define BYTE	unsigned char
	#endif
	
	#ifndef wsnchar
		typedef unsigned char wsnchar;
	#endif
	
	#ifndef BOOL
		typedef unsigned int BOOL;
	#endif
	
	#ifndef DWORD
		typedef unsigned int DWORD;
	#endif
	
	#ifndef LPDWORD
		typedef unsigned int *LPDWORD;
	#endif
	
	#ifndef ULONG
		typedef unsigned long ULONG;
	#endif
	
	#ifndef UCHAR
		#define UCHAR	unsigned char
	#endif
	
	#ifndef WPARAM
		typedef DWORD WPARAM;
	#endif
	
	#ifndef LPARAM
		typedef DWORD LPARAM;
	#endif
	
	#ifndef UINT
		#define UINT	unsigned int
	#endif
	
	#ifndef _SIZE_T
		#define _SIZE_T
		typedef unsigned int size_t;
	#endif
	
	#ifndef TimerSignature
		#define TimerSignature							0x524D4954			// RMIT
	#endif
	
	#ifndef SemaphoreSignature
		#define SemaphoreSignature						0x504D4553			// PMES
	#endif
	
	#ifndef AllocSignature
		#define AllocSignature							0x54524C41			// TRLA
	#endif
	
	typedef void * MUTEX;
	typedef void * SEMAPHORE;
	typedef unsigned long ERROR;
	
	typedef void (* FLM_EXIT_FUNC)( void);
	
	#define DOSNameSpace									0
	#define MACNameSpace									1
	#define MacNameSpace									MACNameSpace
	#define NFSNameSpace									2
	#define FTAMNameSpace								3
	#define OS2NameSpace									4
	#define LONGNameSpace								4
	#define NTNameSpace									5
	#define MAX_NAMESPACES								6
	
	#define NO_RIGHTS_CHECK_ON_OPEN_BIT				0x00010000
	#define ALLOW_SECURE_DIRECTORY_ACCESS_BIT		0x00020000
	#define READ_ACCESS_BIT								0x00000001
	#define WRITE_ACCESS_BIT							0x00000002
	#define DENY_READ_BIT								0x00000004
	#define DENY_WRITE_BIT								0x00000008
	#define COMPATABILITY_MODE_BIT					0x00000010
	#define FILE_WRITE_THROUGH_BIT					0x00000040
	#define FILE_READ_THROUGH_BIT						0x00000080
	#define ALWAYS_READ_AHEAD_BIT						0x00001000
	#define NEVER_READ_AHEAD_BIT						0x00002000
	
	#define READ_ONLY_BIT								0x00000001
	#define HIDDEN_BIT									0x00000002
	#define SYSTEM_BIT									0x00000004
	#define EXECUTE_BIT									0x00000008
	#define SUBDIRECTORY_BIT							0x00000010
	#define ARCHIVE_BIT									0x00000020
	//		  EXECUTE_CONFIRM_BIT						0x00000040
	#define SHAREABLE_BIT								0x00000080 // Valid only on files
	#define OLD_PRIVATE_BIT								0x00000080 // Valid only on directories
	//		  LOW_SEARCH_BIT								0x00000100
	//		  MID_SEARCH_BIT								0x00000200
	//		  HI_SEARCH_BIT 								0x00000400
	#define NO_SUBALLOC_BIT								0x00000800
	#define SMODE_BITS									0x00000700 // search bits
	#define TRANSACTION_BIT		  						0x00001000
	//      OLD_INDEXED_BIT								0x00002000
	#define READ_AUDIT_BIT								0x00004000
	#define WRITE_AUDIT_BIT		  						0x00008000
	#define IMMEDIATE_PURGE_BIT	  					0x00010000
	#define RENAME_INHIBIT_BIT							0x00020000
	#define DELETE_INHIBIT_BIT							0x00040000
	#define COPY_INHIBIT_BIT							0x00080000
	#define FILE_AUDITING_BIT							0x00100000 // system auditing
	#define REMOTE_DATA_ACCESS_BIT					0x00400000 // ie. Data Migration (file only) 
	#define REMOTE_DATA_INHIBIT_BIT					0x00800000 // ie. Data Migration (file only)
	#define REMOTE_DATA_SAVE_KEY_BIT					0x01000000 // ie. Data Migration (file only)
	#define COMPRESS_FILE_IMMEDIATELY_BIT			0x02000000 // immediately try to compress this file (or all files within this subdirectory)
	#define DATA_STREAM_IS_COMPRESSED_BIT			0x04000000 // per data stream directory entry
	#define DO_NOT_COMPRESS_FILE_BIT					0x08000000 // don't compress this file ever (or default files within this subdirectory)
	#define CANT_COMPRESS_DATA_STREAM_BIT			0x20000000 // can't save any space by compressiong this data stream
	#define ATTR_ARCHIVE_BIT							0x40000000 // Object Archive Bit  (EAs, OwnerID, Trustees
	#define ZFS_VOLATILE_BIT							0x80000000 // USED BY NSS (Jim A. Nicolet 11-6-2000)
	
	#define VOLUME_AUDITING_BIT	  					0x01	// system auditing
	#define SUB_ALLOCATION_ENABLED_BIT				0x02	// sub allocation units valid on this volume
	#define FILE_COMPRESSION_ENABLED_BIT			0x04	// file compression enabled on this volume
	#define DATA_MIGRATION_ENABLED_BIT				0x08	// data migration is allowed on this volume
	#define NEW_TRUSTEE_COUNT_BIT						0x10	// .2 Volumes have only 4 trustee entries per volume instead of 6
	#define DIR_SVCS_OBJ_UPGRADED_BIT				0x20	// Modify 3.2 volume DirObjId to new position
	#define VOLUME_IMMEDIATE_PURGE_ENABLED_BIT	0x40	// Volume is marked as immediate purge
	
	// define the data stream values
	
	#define PrimaryDataStream						0
	#define MACResourceForkDataStream			1
	#define FTAMDataStream							2
	
	#define DefinedAccessRightsBits				0x01FB		// all the bits currently used
	#define MaximumDirectoryAccessBits			0x01FF		// all the defined bits for access privileges
	#define AllValidAccessBits						0x100001FF  // all the bits that are valid in CreateDirectory
	
	struct LoadDefinitionStructure
	{
		struct LoadDefinitionStructure *LDLink;
		struct LoadDefinitionStructure *LDKillLink;
		struct LoadDefinitionStructure *LDScanLink;
		struct ResourceTagStructure	*LDResourceList;
		LONG LDIdentificationNumber;
		LONG LDCodeImageOffset;
		LONG LDCodeImageLength;
		LONG LDDataImageOffset;
		LONG LDDataImageLength;
		LONG LDUninitializedDataLength;
		LONG LDCustomDataOffset;
		LONG LDCustomDataSize;
		LONG LDFlags;
		LONG LDType;
		LONG (*LDInitializationProcedure)(
				struct LoadDefinitionStructure *LoadRecord,
				struct ScreenStruct *screenID,
				BYTE *CommandLine,
				BYTE *loadDirectoryPath,
				LONG uninitializedDataLength,
				LONG fileHandle,
				LONG (*ReadRoutine)(
						LONG fileHandle,
						LONG offset,
						void *buffer,
						LONG numberOfBytes),
				LONG customDataOffset,
				LONG customDataSize);
		void (*LDExitProcedure)(void);
		LONG (*LDCheckUnloadProcedure)(
				struct ScreenStruct *screenID);
		void *LDPublics;
		BYTE LDFileName[36];
		BYTE LDName[128];
		LONG *LDCLIBLoadStructure;
		LONG *LDNLMDebugger;
		LONG LDParentID;
		LONG LDReservedForCLIB;
		void *AllocMemory;
		LONG LDTimeStamp;
		void *LDModuleObjectHandle;
		LONG LDMajorVersion;
		LONG LDMinorVersion;
		LONG LDRevision;
		LONG LDYear;
		LONG LDMonth;
		LONG LDDay;
		BYTE *LDCopyright;
		LONG LDSuppressUnloadAllocMsg;
		LONG Reserved2;
		LONG Reserved3;
		LONG Reserved4[64];
		LONG Reserved5[12];
		LONG Reserved6;
		void *LDDomainID;
		struct LoadDefinitionStructure *LDEnvLink;
		void *LDAllocPagesListHead;
		void *LDTempPublicList;
		LONG LDMessageLanguage;
		BYTE **LDMessages;
		LONG LDMessageCount;
		BYTE *LDHelpFile;
		LONG LDMessageBufferSize;
		LONG LDHelpBufferSize;
		LONG LDSharedCodeOffset;
		LONG LDSharedCodeLength;
		LONG LDSharedDataOffset;
		LONG LDSharedDataLength;
		LONG (*LDSharedInitProcedure)(
				struct LoadDefinitionStructure *LoadRecord,
				struct ScreenStruct *screenID,
				BYTE *CommandLine);
		void (*LDSharedExitProcedure)(void);
		LONG LDRPCDataTable;
		LONG LDRealRPCDataTable;
		LONG LDRPCDataTableSize;
		LONG LDNumberOfReferencedPublics;
		void **LDReferencedPublics;
		LONG LDNumberOfReferencedExports;
		LONG LDNICIObject;
		LONG LDAllocPagesListLocked;
		void *LDAddressSpace;
		LONG Reserved7;
	
		void *MPKStubAddress;
		LONG MPKStubSize;
		LONG LDBuildNumber;
		void *LDExtensionData;
	};
	
	typedef struct LoadDefinitionStructure LoadDefStruct;
	
	// defines for the LoadDefinitonStructure's LDFlags member
	
	#define LDModuleIsReEntrantBit				0x00000001
	#define LDModuleCanBeMultiplyLoadedBit		0x00000002
	#define LDSynchronizeStart						0x00000004
	#define LDPseudoPreemptionBit					0x00000008
	#define LDLoadInKernel							0x00000010
	#define Available_0								0x00000020
	#define LDAutoUnload  							0x00000040
	#define LDHiddenModule							0x00000080
	#define LDDigitallySignedFile					0x00000100
	#define LDLoadProtected							0x00000200
	#define LDSharedLibraryModule					0x00000400
	#define LDRestartable							0x00000800
	#define LDUnsafeToUnloadNow					0x00001000
	#define LDModuleIsUniprocessor				0x00002000
	#define LDPreemptable							0x00004000
	#define LDHasSystemCalls						0x00008000
	#define LDVirtualMemory							0x00010000
	#define LDAllExportsMTSafe						0x00020000
	
	typedef struct ARG_DATA_Tag
	{
		char **	ppszArgV;
		char *	pszArgs;
		char *	pszThreadName;
		int		iArgC;
		struct LoadDefinitionStructure *		moduleHandle;
	} ARG_DATA;
	
	// NOTE: It is ok for these header files to be included in the fpackon.h
	// because they should be one-byte packing.
	
	extern "C"
	{
		typedef void * SPINLOCK;
	
		#include "fileio.h"
		#include "lfsproto.h"
	
		#ifndef __DSTRUCT_H__
			#include <dstruct.h>
		#endif
	
		#define needssyncclockprototypes
		#include "synclock.h"
	}
	
	#define MAX_NETWARE_VOLUME_NAME				16
	#define F_NW_DEFAULT_VOLUME_NUMBER			0
	extern FLMBYTE										F_NW_Default_Volume_Name[];
	
	extern RCODE MapNWtoFlaimError(
		LONG	lResult,
		RCODE	defaultRc);
	
	extern LONG FlaimToNWOpenFlags(
		FLMUINT		uiAccess,
		FLMBOOL		bDoDirectIo);
	
	extern "C"
	{
		void kYieldThread( void);
	
		int kGetThreadName( 
			FLMUINT32 ui32ThreadId,
			char * szName, 
			int iBufSize);
	
		void * kCreateThread(
				BYTE *	name,
				void *	(*StartAddress)(void *, void *),
				void *	StackAddressHigh,
				LONG		StackSize,
				void *	Argument);
	
		int kSetThreadName(
			void *		ThreadHandle,
			BYTE *		buffer);
	
		LONG kScheduleThread(
			void *		ThreadHandle);
	
		ERROR kDelayThread( 
			UINT			uiMilliseconds);
	
		void * kCurrentThread(void);
	
		int kDestroyThread(
			void *	ThreadHandle);
	
		void kExitThread(
			void *	ExitStatus);
	
		LONG kSetThreadLoadHandle(
			void *	ThreadHandle,
			LONG		nlmHandle);
	
		LONG GetRunningProcess( void);
	
		void KillMe(
			struct LoadDefinitionStructure *LoadRecord);
	
		void NWYieldIfTime( void);
	
		void CSetD( 
			LONG value, 
			void *address,
			LONG numberOfDWords);
	
		void CMovB( 
			void *src, 
			void *dst,
			LONG numberOfBytes);
	
		void * Alloc( 
			LONG		numberOfBytes,
			LONG		lRTag);
		
		void Free( 
			void *	address);
	
		LONG AllocateResourceTag(
			LONG pvLoadRecord,
			BYTE *pvResourceDescriptionString,
			LONG ResourceSignature);
	
		LONG ReturnResourceTag(
			LONG		RTag,
			BYTE		displayErrorsFlag);
	
		extern LONG ConvertPathString(
			LONG stationNumber,
			BYTE base,
			BYTE *modifierString,
			LONG *volumeNumber,
			LONG *pathBase,
			BYTE *pathString,
			LONG *pathCount);
	
		extern LONG GetEntryFromPathStringBase(
			LONG Station,
			LONG Volume,
			LONG PathBase,
			BYTE *PathString,
			LONG PathCount,
			LONG SourceNameSpace,
			LONG DesiredNameSpace,
			struct DirectoryStructure **Dir,
			LONG *DirectoryNumber);
	
		extern LONG	NDSCreateStreamFile(
			LONG Station,
			LONG Task,
			BYTE *fileName,
			LONG CreateAttributes,
			LONG *fileHandle,
			LONG *DOSDirectoryBase);
	
		extern LONG	NDSOpenStreamFile(
			LONG Station,
			LONG Task,
			BYTE *fileName,
			LONG RequestedRights,
			LONG *fileHandle,
			LONG *DOSDirectoryBase);
	
		extern LONG	NDSDeleteStreamFile(
			LONG Station,
			LONG Task,
			BYTE *fileName,
			LONG *DOSDirectoryBase);
	
		extern LONG ReadFile(
			LONG stationNumber,
			LONG handle,
			LONG startingOffset,
			LONG bytesToRead,
			LONG *actualBytesRead,
			void *buffer);
	
		extern LONG WriteFile(
			LONG stationNumber,
			LONG handle,
			LONG startingOffset,
			LONG bytesToWrite,
			void *buffer);
	
		extern LONG SetFileSize(
			LONG station,
			LONG handle,
			LONG filesize,
			LONG truncateflag);
	
		extern LONG GetFileSize(
			LONG stationNumber,
			LONG handle,
			LONG *fileSize);
	
		extern LONG DirectReadFile(
			LONG station,
			LONG handle,
			LONG startingsector,
			LONG sectorcount,
			BYTE *buffer);
	
		extern LONG SwitchToDirectFileMode(
			LONG station,
			LONG handle);
	
		extern LONG ReturnVolumeMappingInformation(
			LONG volumenumber,
			struct VolumeInformationStructure *volumeInformation);
	
		extern LONG ExpandFileInContiguousBlocks(
			LONG station,
			LONG handle,
			LONG fileblocknumber,
			LONG numberofblocks,
			LONG vblocknumber,
			LONG segnumber);
	
		extern LONG FreeLimboVolumeSpace(
			LONG volumenumber,
			LONG numberofblocks);
	
		extern LONG DirectWriteFileNoWait(
			LONG station,
			LONG handle,
			LONG startingsector,
			LONG sectorcount,
			BYTE *buffer,
			void (*callbackroutine)(LONG, LONG, LONG),
			LONG callbackparameter);
	
		extern LONG DirectWriteFile(
			LONG station,
			LONG handle,
			LONG startingsector,
			LONG sectorcount,
			BYTE *buffer);
	
		extern LONG RevokeFileHandleRights(
			LONG Station,
			LONG Task,
			LONG FileHandle,
			LONG QueryFlag, // 0 = revoke, 1 = query, 2 = revoke and close if last
			LONG removeRights,
			LONG *newRights);
	
		#define DFSFailedCompletion           -1
		#define DFSNormalCompletion           0
		#define DFSInsufficientSpace          1
		#define DFSVolumeSegmentDeactivated   4
		#define DFSTruncationFailure          16
		#define DFSHoleInFileError            17
		#define DFSParameterError             18
		#define DFSOverlapError               19
		#define DFSSegmentError               20
		#define DFSBoundryError               21
		#define DFSInsufficientLimboFileSpace 22
		#define DFSNotInDirectFileMode        23
		#define DFSOperationBeyondEndOfFile   24
		#define DFSOutOfHandles               129
		#define DFSHardIOError                131
		#define DFSInvalidFileHandle          136
		#define DFSNoReadPrivilege            147
		#define DFSNoWritePrivilege           148
		#define DFSFileDetached               149
		#define DFSInsufficientMemory         150
		#define DFSInvalidVolume              152
		#define DFSIOLockError                162
	
		struct VolumeInformationStructure
		{
			LONG VolumeAllocationUnitSizeInBytes;
			LONG VolumeSizeInAllocationUnits;
			LONG VolumeSectorSize;
			LONG AllocationUnitsUsed;
			LONG AllocationUnitsFreelyAvailable;
			LONG AllocationUnitsInDeletedFilesNotAvailable;
			LONG AllocationUnitsInAvailableDeletedFiles;
			LONG NumberOfPhysicalSegmentsInVolume;
			LONG PhysicalSegmentSizeInAllocationUnits[64];
		};
	
		#define MModifyNameBit                 0x0001
		#define MFileAttributesBit             0x0002
		#define MCreateDateBit                 0x0004
		#define MCreateTimeBit                 0x0008
		#define MOwnerIDBit                    0x0010
		#define MLastArchivedDateBit           0x0020
		#define MLastArchivedTimeBit           0x0040
		#define MLastArchivedIDBit             0x0080
		#define MLastUpdatedDateBit            0x0100
		#define MLastUpdatedTimeBit            0x0200
		#define MLastUpdatedIDBit              0x0400
		#define MLastAccessedDateBit           0x0800
		#define MInheritanceRestrictionMaskBit 0x1000
		#define MMaximumSpaceBit               0x2000
		#define MLastUpdatedInSecondsBit       0x4000
	
		struct ModifyStructure
		{
			BYTE *MModifyName;
			LONG  MFileAttributes;
			LONG  MFileAttributesMask;
			WORD  MCreateDate;
			WORD  MCreateTime;
			LONG  MOwnerID;
			WORD  MLastArchivedDate;
			WORD  MLastArchivedTime;
			LONG  MLastArchivedID;
			WORD  MLastUpdatedDate;
			WORD  MLastUpdatedTime;
			LONG  MLastUpdatedID;
			WORD  MLastAccessedDate;
			WORD  MInheritanceGrantMask;
			WORD  MInheritanceRevokeMask;
			int   MMaximumSpace;
			LONG  MLastUpdatedInSeconds;
		};
	
		LONG ImportPublicSymbol(
			LONG		moduleHandle,
			BYTE *	symbolName);
	
		LONG UnImportPublicSymbol(
			LONG		moduleHandle,
			BYTE *	symbolName);
	
		LONG ExportPublicSymbol(
			LONG		moduleHandle,
			BYTE *	symbolName,
			LONG		address);
	
		void SynchronizeStart( void);
	
		LONG CFindLoadModuleHandle( void *);
	
		int atexit(
			FLM_EXIT_FUNC	fnExit);
	
		#define LO_RETURN_HANDLE        0x00000040
	
		LONG LoadModule(
				void *		screenID,
				BYTE *		fileName,
				LONG			loadOptions);
	
		LONG UnloadModule( 
				void **			pScreenID, 
				const char *	commandline);
	
		typedef struct
		{
			FLMUINT32	time_low;
			FLMUINT16	time_mid;
			FLMUINT16	time_hi_and_version;
			FLMBYTE		clk_seq_hi_res;
			FLMBYTE		clk_seq_low;
			FLMBYTE		node[6];
		} NWGUID;
	
		int SGUIDCreate( NWGUID *guidBfr);
	
		void * GetSystemConsoleScreen( void);
	
		LONG SizeOfAllocBlock( 
			void *			AllocAddress);
	
		SEMAPHORE kSemaphoreAlloc(
			BYTE *			pSemaName,
			UINT				SemaCount);
	
		ERROR kSemaphoreFree(
			SEMAPHORE		SemaHandle);
	
		ERROR kSemaphoreWait(
			SEMAPHORE		SemaHandle);
	
		ERROR kSemaphoreTimedWait(
			SEMAPHORE		SemaHandle, 
			UINT				MilliSecondTimeOut);
	
		ERROR kSemaphoreSignal(
			SEMAPHORE		SemaHandle);
	
		UINT kSemaphoreExamineCount(
			SEMAPHORE		SemaHandle);
	
		MUTEX kMutexAlloc(
			BYTE *			MutexName);
	
		ERROR kMutexFree(
			MUTEX				MutexHandle);
	
		ERROR kMutexLock(
			MUTEX				MutexHandle);
	
		ERROR kMutexUnlock(
			MUTEX				MutexHandle);
	
		void CMoveFast( 
			void *			src,
			void *			dst,
			LONG				numberOfBytes);
	
		void EnterDebugger( void);
	
		void GetClosestSymbol(
			BYTE *	szBuffer,
			LONG		udAddress);
	
		LONG GetCurrentTime( void);
	
		void ConvertTicksToSeconds(
			LONG		ticks,
			LONG *	seconds,
			LONG *	tenthsOfSeconds);
	
		void ConvertSecondsToTicks(
			LONG		seconds,
			LONG		tenthsOfSeconds,
			LONG *	ticks);
	
		LONG GetCacheBufferSize(void);
	
		LONG GetOriginalNumberOfCacheBuffers(void);
	
		LONG GetCurrentNumberOfCacheBuffers(void);
	
		LONG GetNLMAllocMemoryCounts(
			FLMUINT		moduleHandle,
			FLMUINT *	freeBytes,
			FLMUINT *	freeNodes,
			FLMUINT *	allocatedBytes,
			FLMUINT *	allocatedNodes,
			FLMUINT *	totalMemory);
	
		LONG atomic_xchg( volatile LONG * address, LONG value);
	
		#define nlm_AtomicExchange( piTarget, iValue) \
			((FLMINT32)atomic_xchg( (volatile LONG *)(piTarget), (LONG)(iValue)))
	
		FLMINT32 nlm_AtomicIncrement( 
			volatile LONG *	piTarget);
	
		FLMINT32 nlm_AtomicDecrement( 
			volatile LONG *	piTarget);
	
		#if !defined( __MWERKS__)
			#pragma aux nlm_AtomicIncrement parm [ecx];
			#pragma aux nlm_AtomicIncrement = \
			0xB8 0x01 0x00 0x00 0x00   		/*  mov	eax, 1  	 			*/ \
			0xF0 0x0F 0xC1 0x01					/*  lock xadd [ecx], eax 	*/ \
			0x40										/*  inc	eax 					*/ \
			parm [ecx]	\
			modify exact [eax];
	
			#pragma aux nlm_AtomicDecrement parm [ecx];
			#pragma aux nlm_AtomicDecrement = \
			0xB8 0xFF 0xFF 0xFF 0xFF   		/*  mov	eax, 0ffffffffh	*/ \
			0xF0 0x0F 0xC1 0x01					/*  lock xadd [ecx], eax 	*/ \
			0x48										/*  dec	eax 					*/ \
			parm [ecx]	\
			modify exact [eax];
		#else
	
			FINLINE FLMINT32 nlm_AtomicIncrement(
				volatile LONG *	piTarget)
			{
				FLMINT32				i32Result;
	
				__asm
				{
					mov	eax, 1
					mov	ecx, piTarget
					lock xadd [ecx], eax
					inc	eax
					mov	i32Result, eax
				}
	
				return( i32Result);
			}
	
			FINLINE FLMINT32 nlm_AtomicDecrement(
				volatile LONG *	piTarget)
			{
				FLMINT32				i32Result;
	
				__asm
				{
					mov	eax, 0ffffffffh
					mov	ecx, piTarget
					lock xadd [ecx], eax
					dec	eax
					mov	i32Result, eax
				}
	
				return( i32Result);
			}
	
		#endif
	
	}	// extern "C"

	#define FSTATIC

	// The typedef for va_list in stdarg.h do not function properly when
	// a va_list is passed down multiple layers as a pointer (va_list *).
	// Therefore, the following definitions/typedefs were taken from a
	// "fixed" version of stdarg.h implemented by DS.

	typedef unsigned long f_va_list;

	#define f_argsize(x) \
		((sizeof(x)+sizeof(int)-1) & ~(sizeof(int)-1))

	#define f_va_start(ap, parmN) \
		((void)((ap) = (unsigned long)&(parmN) + f_argsize(parmN)))

	#define f_va_arg(ap, type) \
		(*(type *)(((ap) += f_argsize(type)) - (f_argsize(type))))

	#define f_va_end(ap)	\
		((void)0)

	FINLINE char * f_strcpy(
		char *			d,
		const char *	s)
	{
		while ((*d++ = *s++) != 0);
		return( d);
	}

	FINLINE unsigned f_strlen(
		const char *	s)
	{
		const char *	b = s;

		while (*s)
		{
			s++;
		}

		return( s - b);
	}

	FINLINE int f_strcmp(
		const char *		s1,
		const char *		s2)
	{
		while( *s1 == *s2 && *s1)
		{
			s1++;
			s2++;
		}
		return( (int)(*s1 - *s2));
	}

	FINLINE char * f_strncpy(
		char *			dest,
		const char *	src,
		unsigned			n)
	{
		while( n)
		{
			*dest++ = *src;
			if( *src)
			{
				src++;
			}
			n--;
		}

		*dest = 0;
		return( dest);
	}

	FINLINE int f_strncmp(
		const char *	s1,
		const char *	s2,
		unsigned			n)
	{
		while( *s1 == *s2 && *s1 && n)
		{
			s1++;
			s2++;
			n--;
		}

		if( n)
		{
			return( (*s1 - *s2));
		}

		return( (int)0);
	}

	char * f_strstr(
		const char *		pszStr1,
		const char *		pszStr2);

	FLMINT f_stricmp(
		const char *		pszStr1,
		const char *		pszStr2);

	FLMINT f_strnicmp(
		const char *		pszStr1,
		const char *		pszStr2,
		FLMINT				iLen);

	FINLINE char * f_strcat(
		char *				dst,
		const char *		src)
	{	
		const char *	p = src;
		char * 			q = dst;
		
		while (*q++);
		q--;
		while( (*q++ = *p++) != 0);
		
		return(dst);
	}

	FINLINE char * f_strncat(
		char *			dst ,
		const char *	src,
		unsigned			n)
	{
		const char *		p = src;
		char *				q = dst;
		
		while (*q++);
		
		q--; n++;
		
		while( --n)
		{
			if( (*q++ = *p++) == 0)
			{
				q--;
				break;
			}
		}
		
		*q = 0;
		return( dst);
	}

	char * f_strupr(
		char *		pszStr);

	FLMINT f_memcmp(
		const void *	pvMem1,
		const void *	pvMem2,
		FLMUINT			uiSize);

	#define f_memcpy( dest, src, size) \
		CMoveFast( (void *)(src), (void *)(dest), size)

	void * f_memset(
		void *			pvMem,
		FLMBYTE			ucByte,
		FLMUINT			uiSize);

	#define f_memset( m, c, size) \
		f_memset( (void *)(m), c, size)

	void * f_memmove(
		void *			pvDest,
		const void *	pvSrc,
		FLMUINT			uiSize);

#elif defined( FLM_WIN)

	#define FSTATIC		static
	
	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	
	#ifndef WIN32_EXTRA_LEAN
		#define WIN32_EXTRA_LEAN
	#endif
	
	// This pragma is needed because FLAIM may be built with a
	// packing other than 8-bytes on Win (such as 1-byte packing).
	// Code in FLAIM that uses windows structures and system calls
	// MUST use 8-byte packing (the packing used by the O/S).
	// See Microsoft technical article Q117388.

	#pragma pack( push, enter_windows, 8)
		#include <windows.h>
		#include <stdio.h>
		#include <time.h>
		#include <memory.h>
		#include <stdlib.h>
		#include <string.h>
		#include <process.h>
		#include <winsock.h>
		#include <imagehlp.h>
		#include <malloc.h>
		#include <rpc.h>
		#include <process.h>
		#include <stddef.h>
	#pragma pack( pop, enter_windows)

	// Conversion from XXX to YYY, possible loss of data

	#pragma warning( disable : 4244) 

	// Local variable XXX may be used without having been initialized

	#pragma warning( disable : 4701)

	// Function XXX not inlined

	#pragma warning( disable : 4710) 

	#define f_stricmp( str1, str2) \
		_stricmp( (str1), (str2))

	#define f_strnicmp( str1, str2, size) \
		_strnicmp( (str1), (str2),(size_t)(size))

	#define f_memcpy( dest, src, size) \
		memcpy((void  *)(dest), (void  *)(src),(size_t)(size))

	#define f_memmove( dest, src, length) \
		memmove((void  *)(dest), (void  *)(src),(size_t)(length))

	#define f_memset( src, chr, size) \
		memset((void  *)(src),(chr),(size_t)(size))

	#define f_memcmp( str1, str2, length) \
		memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))

	#define f_strcat( dest, src) \
		strcat( (char*)(dest), (char*)(src))

	#define f_strcmp( str1, str2) \
		strcmp( (char*)(str1), (char*)(str2))

	#define f_strcpy( dest, src) \
		strcpy( (char*)(dest), (char*)(src))

	#define f_strncpy( dest, src, length) \
		strncpy( (char*)(dest), (char*)(src), (size_t)(length))

	#define f_strlen( str) \
		((FLMUINT)strlen( (char*)(str)))

	#define f_strncmp( str1, str2, size) \
		strncmp( (char*)(str1), (char*)(str2), (size_t)(size))

	#define f_strrchr( str, value ) \
		strrchr( (char*)(str), (int)value)

	#define f_strstr( str1, str2) \
		strstr( (char*)(str1), (char*)(str2))

	#define f_strncat( str1, str2, n) \
		strncat( (str1), (str2), n)

	#define f_strupr( str) \
		_strupr( (str))

	#define f_va_list			va_list
	#define f_va_start		va_start
	#define f_va_arg			va_arg
	#define f_va_end			va_end
	
#elif defined( FLM_UNIX)

	#include <errno.h>
	#include <glob.h>
	#include <limits.h>
	#include <netdb.h>
	#include <sys/types.h>
	#include <netinet/in.h>
	#include <arpa/nameser.h>
	#include <resolv.h>
	#include <stdarg.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <string.h>
	#include <strings.h>
	#include <time.h>
	#include <unistd.h>
	#include <utime.h>
	#include <arpa/inet.h>
	#include <netinet/tcp.h>
	#include <sys/mman.h>
	#include <sys/resource.h>
	#include <sys/socket.h>
	#include <sys/stat.h>
	#include <sys/time.h>
	
	#ifndef  _POSIX_THREADS
		#define  _POSIX_THREADS
	#endif
	
	#include <pthread.h>

	#ifndef FLM_OSX
		#include <malloc.h>
	#endif
	
	#ifdef FLM_AIX
		#include <sys/atomic_op.h>
	#endif

	#define FSTATIC		static

	#define f_stricmp(str1,str2) \
		strcasecmp( (str1), (str2))

	#define f_strnicmp(str1,str2,size_t) \
		strncasecmp( (str1), (str2),size_t)

	#define f_memcpy( dest, src, size) \
		memcpy( (void*)(dest), (void*)(src), size)

	#define f_memmove( dest, src, len) \
		memmove( (void*)(dest), (void*)(src), len)

	#define f_memset( src, chr, size) \
		memset((void  *)(src),(chr),(size_t)(size))

	#define f_memcmp( str1, str2, length) \
		memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))

	#define f_strcat( dest, src) \
		strcat( (char*)(dest), (char*)(src))

	#define f_strcmp( str1, str2) \
		strcmp( (char*)(str1), (char*)(str2))

	#define f_strcpy( dest, src) \
		strcpy( (char*)(dest), (char*)(src))

	#define f_strncpy( dest, src, length) \
		strncpy( (char*)(dest), (char*)(src), (size_t)(length))

	#define f_strlen( str) \
		strlen( (char*)(str))

	#define f_strncmp( str1, str2, size) \
		strncmp( (char*)(str1), (char*)(str2), (size_t)(size))

	#define f_strrchr( str, value ) \
		strrchr( (char*)(str), (int)value)

	#define f_strstr( str1, str2) \
		strstr( (char*)(str1), (char*)(str2))

	#define f_strncat( str1, str2, n) \
		strncat( (str1), (str2), n)

	char * f_strupr(
		char *		pszStr);

	#define f_strupr( str) \
		f_strupr( (str))

	#define f_va_list			va_list
	#define f_va_start		va_start
	#define f_va_arg			va_arg
	#define f_va_end			va_end

#endif

/****************************************************************************
									CROSS PLATFORM DEFINITIONS
****************************************************************************/

#define F_UNREFERENCED_PARM( parm) \
	(void)parm

#if defined( __va_copy)
	#define  f_va_copy(to, from) \
		__va_copy(to, from)
#else
	#define f_va_copy(to, from) \
		((to) = (from))
#endif

#define shiftN(data,size,distance) \
		f_memmove((FLMBYTE *)(data) + (FLMINT)(distance), \
		(FLMBYTE *)(data), (unsigned)(size))

/****************************************************************************
								FLAIM's Assert Layer
****************************************************************************/

#ifndef FLM_DEBUG
	#define flmAssert( exp)
#else
	#ifdef FLM_DBG_LOG
		void flmDbgLogFlush( void);
	#endif

	#if defined( FLM_WIN)
		#ifdef FLM_DBG_LOG
			#define flmAssert( exp) \
				(void)( (exp) || (flmDbgLogFlush(), DebugBreak(), 0))
		#else
			#define flmAssert( exp) \
				(void)( (exp) || (DebugBreak(), 0))
		#endif

	#elif defined( FLM_NLM)
		#ifdef FLM_DBG_LOG
			#define flmAssert( exp) \
				(void)( (exp) || (flmDbgLogFlush(), EnterDebugger(), 0))
		#else
			#define flmAssert( exp) \
				(void)( (exp) || ( EnterDebugger(), 0))
		#endif

	#elif defined( FLM_UNIX)
		#include <assert.h>
		
		#ifdef FLM_DBG_LOG
			#define flmAssert( exp) \
				(void)( (exp) || (flmDbgLogFlush(), assert(0), 0))
		#else
			#define flmAssert( exp) \
				(void)( (exp) || (assert(0), 0))
		#endif

	#else
		#define flmAssert( exp)	
	#endif

#endif

#ifdef FLM_DEBUG
	#define flmReleaseAssert( exp)	flmAssert( exp)
#else
	#if defined( FLM_WIN)
		#define flmReleaseAssert( exp) \
			(void)( (exp) || (DebugBreak(), 0))

	#elif defined( FLM_NLM)
		#define flmReleaseAssert( exp) \
			(void)( (exp) || ( EnterDebugger(), 0))

	#elif defined( FLM_UNIX)
		#include <assert.h>
		#define flmReleaseAssert( exp) \
			(void)( (exp) || (abort(), 0))
	#else
		#define flmReleaseAssert( exp)
	#endif
#endif


#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

FLMUINT f_breakpoint(
	FLMUINT		uiBreakFlag);

/****************************************************************************
									ASCII Constants
****************************************************************************/
	#define ASCII_TAB						0x09
	#define ASCII_NEWLINE				0x0A
	#define ASCII_CR                 0x0D
	#define ASCII_CTRLZ              0x1A
	#define ASCII_SPACE              0x20
	#define ASCII_DQUOTE					0x22
	#define ASCII_POUND              0x23
	#define ASCII_DOLLAR             0x24
	#define ASCII_SQUOTE             0x27
	#define ASCII_WILDCARD           0x2A
	#define ASCII_PLUS               0x2B
	#define ASCII_COMMA              0x2C
	#define ASCII_DASH               0x2D
	#define ASCII_MINUS              0x2D
	#define ASCII_DOT                0x2E
	#define ASCII_SLASH              0x2F
	#define ASCII_COLON              0x3A
	#define ASCII_SEMICOLON				0x3B
	#define ASCII_EQUAL              0x3D
	#define ASCII_QUESTIONMARK			0x3F
	#define ASCII_AT                 0x40
	#define ASCII_BACKSLASH				0x5C
	#define ASCII_CARAT					0x5E
	#define ASCII_UNDERSCORE			0x5F
	#define ASCII_TILDE					0x7E
	#define ASCII_AMP						0x26

	#define ASCII_UPPER_A				0x41
	#define ASCII_UPPER_B				0x42
	#define ASCII_UPPER_C				0x43
	#define ASCII_UPPER_D				0x44
	#define ASCII_UPPER_E				0x45
	#define ASCII_UPPER_F				0x46
	#define ASCII_UPPER_G				0x47
	#define ASCII_UPPER_H				0x48
	#define ASCII_UPPER_I				0x49
	#define ASCII_UPPER_J				0x4A
	#define ASCII_UPPER_K				0x4B
	#define ASCII_UPPER_L				0x4C
	#define ASCII_UPPER_M				0x4D
	#define ASCII_UPPER_N				0x4E
	#define ASCII_UPPER_O				0x4F
	#define ASCII_UPPER_P				0x50
	#define ASCII_UPPER_Q				0x51
	#define ASCII_UPPER_R				0x52
	#define ASCII_UPPER_S				0x53
	#define ASCII_UPPER_T				0x54
	#define ASCII_UPPER_U				0x55
	#define ASCII_UPPER_V				0x56
	#define ASCII_UPPER_W				0x57
	#define ASCII_UPPER_X				0x58
	#define ASCII_UPPER_Y				0x59
	#define ASCII_UPPER_Z				0x5A

	#define ASCII_LOWER_A				0x61
	#define ASCII_LOWER_B				0x62
	#define ASCII_LOWER_C				0x63
	#define ASCII_LOWER_D				0x64
	#define ASCII_LOWER_E				0x65
	#define ASCII_LOWER_F				0x66
	#define ASCII_LOWER_G				0x67
	#define ASCII_LOWER_H				0x68
	#define ASCII_LOWER_I				0x69
	#define ASCII_LOWER_J				0x6A
	#define ASCII_LOWER_K				0x6B
	#define ASCII_LOWER_L				0x6C
	#define ASCII_LOWER_M				0x6D
	#define ASCII_LOWER_N				0x6E
	#define ASCII_LOWER_O				0x6F
	#define ASCII_LOWER_P				0x70
	#define ASCII_LOWER_Q				0x71
	#define ASCII_LOWER_R				0x72
	#define ASCII_LOWER_S				0x73
	#define ASCII_LOWER_T				0x74
	#define ASCII_LOWER_U				0x75
	#define ASCII_LOWER_V				0x76
	#define ASCII_LOWER_W				0x77
	#define ASCII_LOWER_X				0x78
	#define ASCII_LOWER_Y				0x79
	#define ASCII_LOWER_Z				0x7A

	#define ASCII_ZERO					0x30
	#define ASCII_ONE						0x31
	#define ASCII_TWO						0x32
	#define ASCII_THREE					0x33
	#define ASCII_FOUR					0x34
	#define ASCII_FIVE					0x35
	#define ASCII_SIX						0x36
	#define ASCII_SEVEN					0x37
	#define ASCII_EIGHT					0x38
	#define ASCII_NINE					0x39

#define NATIVE_SPACE             ' '
#define NATIVE_DOT               '.'
#define NATIVE_PLUS              '+'
#define NATIVE_MINUS					'-'
#define NATIVE_WILDCARD				'*'
#define NATIVE_QUESTIONMARK		'?'

#define NATIVE_UPPER_A				'A'
#define NATIVE_UPPER_F				'F'
#define NATIVE_UPPER_X				'X'
#define NATIVE_UPPER_Z				'Z'
#define NATIVE_LOWER_A				'a'
#define NATIVE_LOWER_F				'f'
#define NATIVE_LOWER_X				'x'
#define NATIVE_LOWER_Z				'z'
#define NATIVE_ZERO              '0'
#define NATIVE_NINE              '9'

#define f_stringToAscii(str)

#define f_toascii(native)			(native)

#define f_tonative(ascii)			(ascii)

#define f_toupper(native)			(((native) >= 'a' && (native) <= 'z') \
												? (native) - 'a' + 'A' : (native))

#define f_tolower(native)			(((native) >= 'A' && (native) <= 'Z') \
												? (native) - 'A' + 'a' : (native))

#define f_islower(native)			((native) >= 'a' && (native) <= 'z')

// Unicode character constants

#define FLM_UNICODE_LINEFEED			((FLMUNICODE)10)
#define FLM_UNICODE_SPACE				((FLMUNICODE)32)
#define FLM_UNICODE_BANG				((FLMUNICODE)33)
#define FLM_UNICODE_QUOTE				((FLMUNICODE)34)
#define FLM_UNICODE_POUND				((FLMUNICODE)35)
#define FLM_UNICODE_DOLLAR				((FLMUNICODE)36)
#define FLM_UNICODE_PERCENT			((FLMUNICODE)37)
#define FLM_UNICODE_AMP					((FLMUNICODE)38)
#define FLM_UNICODE_APOS				((FLMUNICODE)39)
#define FLM_UNICODE_LPAREN				((FLMUNICODE)40)
#define FLM_UNICODE_RPAREN				((FLMUNICODE)41)
#define FLM_UNICODE_ASTERISK			((FLMUNICODE)42)
#define FLM_UNICODE_PLUS				((FLMUNICODE)43)
#define FLM_UNICODE_COMMA				((FLMUNICODE)44)
#define FLM_UNICODE_HYPHEN				((FLMUNICODE)45)
#define FLM_UNICODE_PERIOD				((FLMUNICODE)46)
#define FLM_UNICODE_FSLASH				((FLMUNICODE)47)

#define FLM_UNICODE_0					((FLMUNICODE)48)
#define FLM_UNICODE_1					((FLMUNICODE)49)
#define FLM_UNICODE_2					((FLMUNICODE)50)
#define FLM_UNICODE_3					((FLMUNICODE)51)
#define FLM_UNICODE_4					((FLMUNICODE)52)
#define FLM_UNICODE_5					((FLMUNICODE)53)
#define FLM_UNICODE_6					((FLMUNICODE)54)
#define FLM_UNICODE_7					((FLMUNICODE)55)
#define FLM_UNICODE_8					((FLMUNICODE)56)
#define FLM_UNICODE_9					((FLMUNICODE)57)

#define FLM_UNICODE_COLON				((FLMUNICODE)58)
#define FLM_UNICODE_SEMI				((FLMUNICODE)59)
#define FLM_UNICODE_LT					((FLMUNICODE)60)
#define FLM_UNICODE_EQ					((FLMUNICODE)61)
#define FLM_UNICODE_GT					((FLMUNICODE)62)
#define FLM_UNICODE_QUEST				((FLMUNICODE)63)
#define FLM_UNICODE_ATSIGN				((FLMUNICODE)64)

#define FLM_UNICODE_A					((FLMUNICODE)65)
#define FLM_UNICODE_B					((FLMUNICODE)66)
#define FLM_UNICODE_C					((FLMUNICODE)67)
#define FLM_UNICODE_D					((FLMUNICODE)68)
#define FLM_UNICODE_E					((FLMUNICODE)69)
#define FLM_UNICODE_F					((FLMUNICODE)70)
#define FLM_UNICODE_G					((FLMUNICODE)71)
#define FLM_UNICODE_H					((FLMUNICODE)72)
#define FLM_UNICODE_I					((FLMUNICODE)73)
#define FLM_UNICODE_J					((FLMUNICODE)74)
#define FLM_UNICODE_K					((FLMUNICODE)75)
#define FLM_UNICODE_L					((FLMUNICODE)76)
#define FLM_UNICODE_M					((FLMUNICODE)77)
#define FLM_UNICODE_N					((FLMUNICODE)78)
#define FLM_UNICODE_O					((FLMUNICODE)79)
#define FLM_UNICODE_P					((FLMUNICODE)80)
#define FLM_UNICODE_Q					((FLMUNICODE)81)
#define FLM_UNICODE_R					((FLMUNICODE)82)
#define FLM_UNICODE_S					((FLMUNICODE)83)
#define FLM_UNICODE_T					((FLMUNICODE)84)
#define FLM_UNICODE_U					((FLMUNICODE)85)
#define FLM_UNICODE_V					((FLMUNICODE)86)
#define FLM_UNICODE_W					((FLMUNICODE)87)
#define FLM_UNICODE_X					((FLMUNICODE)88)
#define FLM_UNICODE_Y					((FLMUNICODE)89)
#define FLM_UNICODE_Z					((FLMUNICODE)90)

#define FLM_UNICODE_LBRACKET			((FLMUNICODE)91)
#define FLM_UNICODE_BACKSLASH			((FLMUNICODE)92)
#define FLM_UNICODE_RBRACKET			((FLMUNICODE)93)
#define FLM_UNICODE_UNDERSCORE		((FLMUNICODE)95)

#define FLM_UNICODE_a					((FLMUNICODE)97)
#define FLM_UNICODE_b					((FLMUNICODE)98)
#define FLM_UNICODE_c					((FLMUNICODE)99)
#define FLM_UNICODE_d					((FLMUNICODE)100)
#define FLM_UNICODE_e					((FLMUNICODE)101)
#define FLM_UNICODE_f					((FLMUNICODE)102)
#define FLM_UNICODE_g					((FLMUNICODE)103)
#define FLM_UNICODE_h					((FLMUNICODE)104)
#define FLM_UNICODE_i					((FLMUNICODE)105)
#define FLM_UNICODE_j					((FLMUNICODE)106)
#define FLM_UNICODE_k					((FLMUNICODE)107)
#define FLM_UNICODE_l					((FLMUNICODE)108)
#define FLM_UNICODE_m					((FLMUNICODE)109)
#define FLM_UNICODE_n					((FLMUNICODE)110)
#define FLM_UNICODE_o					((FLMUNICODE)111)
#define FLM_UNICODE_p					((FLMUNICODE)112)
#define FLM_UNICODE_q					((FLMUNICODE)113)
#define FLM_UNICODE_r					((FLMUNICODE)114)
#define FLM_UNICODE_s					((FLMUNICODE)115)
#define FLM_UNICODE_t					((FLMUNICODE)116)
#define FLM_UNICODE_u					((FLMUNICODE)117)
#define FLM_UNICODE_v					((FLMUNICODE)118)
#define FLM_UNICODE_w					((FLMUNICODE)119)
#define FLM_UNICODE_x					((FLMUNICODE)120)
#define FLM_UNICODE_y					((FLMUNICODE)121)
#define FLM_UNICODE_z					((FLMUNICODE)122)

#define FLM_UNICODE_LBRACE				((FLMUNICODE)123)
#define FLM_UNICODE_PIPE				((FLMUNICODE)124)
#define FLM_UNICODE_RBRACE				((FLMUNICODE)125)
#define FLM_UNICODE_TILDE				((FLMUNICODE)126)
#define FLM_UNICODE_C_CEDILLA			((FLMUNICODE)199)
#define FLM_UNICODE_N_TILDE			((FLMUNICODE)209)
#define FLM_UNICODE_c_CEDILLA			((FLMUNICODE)231)
#define FLM_UNICODE_n_TILDE			((FLMUNICODE)241)

/****************************************************************************
								WORD/BYTE ORDERING MACROS
****************************************************************************/

FLMUINT32 byteToLong( 
	FLMBYTE *		ptr);

#define  byteToLong(p)  ( \
   ((FLMUINT32) ( ((((FLMBYTE *)(p))[ 0]) << 8) | (((FLMBYTE *)(p))[ 1]) ) << 16 ) | \
	 (FLMUINT16) ( ((((FLMBYTE *)(p))[ 2]) << 8) | (((FLMBYTE *)(p))[ 3]) ) )

FINLINE FLMUINT64 byteToLong64( 
	FLMBYTE *		pucBuf)
{
	FLMUINT64		ui64Val = 0;

	ui64Val |= ((FLMUINT64)pucBuf[ 0]) << 56;
	ui64Val |= ((FLMUINT64)pucBuf[ 1]) << 48;
	ui64Val |= ((FLMUINT64)pucBuf[ 2]) << 40;
	ui64Val |= ((FLMUINT64)pucBuf[ 3]) << 32;
	ui64Val |= ((FLMUINT64)pucBuf[ 4]) << 24;
	ui64Val |= ((FLMUINT64)pucBuf[ 5]) << 16;
	ui64Val |= ((FLMUINT64)pucBuf[ 6]) << 8;
	ui64Val |= ((FLMUINT64)pucBuf[ 7]);

	return( ui64Val);
}

FLMUINT32 byteToInt(
	FLMBYTE *		ptr);

#define  byteToInt(p)  ( \
 	 (FLMUINT16) ( ((((FLMBYTE *)(p))[ 0]) << 8) | (((FLMBYTE *)(p))[ 1]) ) )

void longToByte( 
	FLMINT32			uiNum,
	FLMBYTE *		ptr);

#define longToByte( n, p) { \
	FLMUINT32 ui32Temp = (FLMUINT32) (n); FLMBYTE * pTemp = (FLMBYTE *)(p); \
			pTemp[0] = (FLMBYTE) (ui32Temp >> 24); \
			pTemp[1] = (FLMBYTE) (ui32Temp >> 16); \
			pTemp[2] = (FLMBYTE) (ui32Temp >>  8); \
			pTemp[3] = (FLMBYTE) (ui32Temp      ); \
	}

void long64ToByte( 
	FLMINT64			uiNum,
	FLMBYTE *		ptr);

#define long64ToByte( n, p) { \
	FLMUINT64 ui64Temp = (FLMUINT64) (n); FLMBYTE * pTemp = (FLMBYTE *)(p); \
			pTemp[0] = (FLMBYTE) (ui64Temp >> 56); \
			pTemp[1] = (FLMBYTE) (ui64Temp >> 48); \
			pTemp[2] = (FLMBYTE) (ui64Temp >> 40); \
			pTemp[3] = (FLMBYTE) (ui64Temp >> 32); \
			pTemp[4] = (FLMBYTE) (ui64Temp >> 24); \
			pTemp[5] = (FLMBYTE) (ui64Temp >> 16); \
			pTemp[6] = (FLMBYTE) (ui64Temp >>  8); \
			pTemp[7] = (FLMBYTE) (ui64Temp      ); \
	}

void intToByte( 
	FLMINT16			uiNum,
	FLMBYTE *		ptr);

#define intToByte( n, p) { \
	FLMUINT16 ui16Temp = (FLMUINT16) (n); FLMBYTE * pTemp = (FLMBYTE *) (p); \
			pTemp[0] = (FLMBYTE) (ui16Temp >>  8); \
			pTemp[1] = (FLMBYTE) (ui16Temp      ); \
	}

#ifndef FLM_BIG_ENDIAN

	// Sanity check

	#if defined( FLM_SPARC)
		#error Wrong endian order selected
	#endif

	#define LO(wrd) 	(*(FLMUINT8  *)&wrd)
	#define HI(wrd) 	(*((FLMUINT8 *)&wrd + 1))

	#if( defined( FLM_UNIX) && defined( FLM_STRICT_ALIGNMENT))

		/****************************************************************************
										LITTLE ENDIAN BYTE ORDER
										NOT SINGLE-BYTE ALIGNED
		****************************************************************************/

		#define FB2UW( bp)			((FLMUINT16)((((FLMUINT16)(((FLMUINT8 *)(bp))[1]))<<8) | \
						 					(((FLMUINT16)(((FLMUINT8 *)(bp))[0])))))

		#define FB2UD( bp)			((FLMUINT32)(	(((FLMUINT32)(((FLMUINT8 *)(bp))[3]))<<24) | \
											(((FLMUINT32)(((FLMUINT8 *)(bp))[2]))<<16) | \
											(((FLMUINT32)(((FLMUINT8 *)(bp))[1]))<< 8) | \
											(((FLMUINT32)(((FLMUINT8 *)(bp))[0])))))

		#define UW2FBA( uw, bp)		(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)(uw)), \
											 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)((((uw) & 0xff00)>>8))))

		#define UD2FBA( udw, bp)	(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((udw) & 0xff)), \
											 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((udw) & 0xff00)>>8)), \
											 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((udw) & 0xff0000)>>16)), \
											 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((udw) & 0xff000000)>>24)))

	#else

		/****************************************************************************
										LITTLE ENDIAN BYTE ORDER
										SINGLE-BYTE ALIGNED
		****************************************************************************/

		#define FB2UW( fbp)					(*((FLMUINT16 *)(fbp)))
		#define FB2UD( fbp)					(*((FLMUINT32 *)(fbp)))
		#define UW2FBA( uw, fbp)			(*((FLMUINT16 *)(fbp)) = ((FLMUINT16) (uw)))
		#define UD2FBA( uw, fbp)			(*((FLMUINT32 *)(fbp)) = ((FLMUINT32) (uw)))

	#endif
   
#else

	/****************************************************************************
									BIG ENDIAN BYTE ORDER (UNIX) 
									NOT SINGLE-BYTE ALIGNED	
	****************************************************************************/

	#if defined( __i386__)
		#error Wrong endian order selected
	#endif

	#define	LO(wrd) 				(*((FLMUINT8 *)&wrd + 1))
	#define	HI(wrd) 				(*(FLMUINT8  *)&wrd)

	#define	FB2UW( bp )			( (FLMUINT16)(	(((FLMUINT16)(((FLMUINT8 *)(bp))[1]))<<8) | \
										(((FLMUINT16)(((FLMUINT8 *)(bp))[0]))   ) ))

	#define	FB2UD( bp )			( (FLMUINT32)(	(((FLMUINT32)(((FLMUINT8 *)(bp))[3]))<<24) | \
										(((FLMUINT32)(((FLMUINT8 *)(bp))[2]))<<16) | \
										(((FLMUINT32)(((FLMUINT8 *)(bp))[1]))<< 8) | \
										(((FLMUINT32)(((FLMUINT8 *)(bp))[0]))    ) ))

	#define	UW2FBA( uw, bp )	(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)(uw)), \
										 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)((((uw) & 0xff00)>>8))))

	#define	UD2FBA( udw, bp)	(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((udw) & 0xff)), \
										 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((udw) & 0xff00)>>8)), \
										 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((udw) & 0xff0000)>>16)), \
										 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((udw) & 0xff000000)>>24)))

#endif

/****************************************************************************
								File Path Functions & Macros
****************************************************************************/

#if defined( FLM_WIN) || defined( FLM_NLM)
	#define FWSLASH     '/'
	#define SLASH       '\\'
	#define SSLASH      "\\"
	#define COLON       ':'
	#define PERIOD      '.'
	#define PARENT_DIR  ".."
	#define CURRENT_DIR "."
#else
	#ifndef FWSLASH
		#define FWSLASH '/'
	#endif

	#ifndef SLASH
		#define SLASH  '/'
	#endif

	#ifndef SSLASH
		#define SSLASH      "/"
	#endif

	#ifndef COLON
		#define COLON  ':'
	#endif

	#ifndef PERIOD
		#define PERIOD '.'
	#endif

	#ifndef PARENT_DIR
		#define PARENT_DIR ".."
	#endif

	#ifndef CURRENT_DIR
		#define CURRENT_DIR "."
	#endif
#endif

/****************************************************************************
								CPU Release Functions										
****************************************************************************/

#ifdef FLM_NLM
	#define f_yieldCPU()	\
		NWYieldIfTime()
#else
	#define f_yieldCPU()
#endif

void f_sleep(
	FLMUINT		uiMilliseconds);

#ifdef FLM_WIN
	#define f_sleep( uiMilliseconds) \
		Sleep( (DWORD)uiMilliseconds)
#endif

/*****************************************************************************
					 					Mutexes
*****************************************************************************/

#ifndef FLM_NLM
	RCODE f_mutexCreate(
		F_MUTEX *	phMutex);

	void f_mutexDestroy(
		F_MUTEX *	phMutex);
#endif

#if defined( FLM_NLM)

	FINLINE RCODE f_mutexCreate(
		F_MUTEX *	phMutex)
	{
		if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"NOVDB")) == F_MUTEX_NULL)
		{
			return( RC_SET( FERR_MEM));
		}

		return( FERR_OK);
	}

	FINLINE void f_mutexDestroy(
		F_MUTEX *	phMutex)
	{
		if (*phMutex != F_MUTEX_NULL)
		{
			(void)kMutexFree( (MUTEX)(*phMutex));
			*phMutex = F_MUTEX_NULL;
		}
	}

	FINLINE void f_mutexLock( 
		F_MUTEX		hMutex)
	{
		(void)kMutexLock( (MUTEX)hMutex);
	}

	FINLINE void f_mutexUnlock(
		F_MUTEX		hMutex)
	{
		(void)kMutexUnlock( (MUTEX)hMutex);
	}

#elif defined( FLM_WIN)

	FINLINE void f_mutexLock(
		F_MUTEX		hMutex)
	{
		(void)EnterCriticalSection( (CRITICAL_SECTION *)hMutex);
	}

	FINLINE void f_mutexUnlock(
		F_MUTEX		hMutex)
	{
		(void)LeaveCriticalSection( (CRITICAL_SECTION *)hMutex);
	}
	
#elif defined( FLM_UNIX)

	void f_mutexLock(
		F_MUTEX		hMutex);
	
	void f_mutexUnlock(
		F_MUTEX		hMutex);

#endif


/*****************************************************************************
					 					Semaphores
*****************************************************************************/

/* pass this define to semwait if you want to wait forever.  may cause hung */
/* machines or proccesses                                                                                                                                       */
#define F_SEM_WAITFOREVER			(0xFFFFFFFF)

#if defined( FLM_WIN)
	typedef HANDLE					F_SEM;
	typedef HANDLE *				F_SEM_p;
	#define F_SEM_NULL			NULL

#elif defined( FLM_NLM)
	typedef SEMAPHORE				F_SEM;
	typedef SEMAPHORE *			F_SEM_p;
	#define F_SEM_NULL			0

#elif defined( FLM_UNIX)

	/* Added by R. Ganesan because Event Semaphores are not the same as
		Mutex Semaphores. Event Semaphores can be signalled without being
		locked. Event Semaphores need to have a genuine wait till they are
		signalled.

		Note: If semaphore.h is not available; this can be implemented in
		terms of condition variables. Condition variables can also be used
		if it is desired that multiple signals == one signal.
	*/

	#if defined( FLM_AIX) || defined( FLM_OSX)
	
	// OS X only has named semaphores, not unamed ones.  If does, however
	// have condition variables and mutexes, so we'll just use the AIX
	// code (and get timed waits as a bonus...)

		typedef struct
		{
			pthread_mutex_t lock;
			pthread_cond_t  cond;
			int             count;
		} sema_t;

		int sema_init( sema_t * sem);
		void sema_destroy( sema_t * sem);
		void p_operation_cleanup( void * arg);
		int sema_wait( sema_t * sem);
		int sema_timedwait( sema_t * sem, unsigned int uiTimeout);
		int sema_signal( sema_t * sem);

	#else
		#include <semaphore.h>
		
		// Note for future reference: We had problems in the AIX build for
		// eDir 8.8 with open being redefined to open64 in some places
		// because we needed support for large files and this was causing
		// problems with FlmBlobImp::open().  The redefinition happens in
		// fcntl.h, and only fposix.cpp needs to include it.  Unfortunately,
		// semaphore.h also includes fcntl.h, and most flaim files end up 
		// including this ftksem.h.  This means that if we ever enable
		// large file support on other unix's, we might bump into these
		// problems again.
		
	#endif
	
	typedef F_SEM *				F_SEM_p;
	#define F_SEM_NULL			NULL

#else
	#error Unsupported platform
#endif

#if defined( FLM_NLM)

	FINLINE RCODE f_semCreate(
		F_SEM *		phSem)
	{
		if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"NOVDB", 0)) == F_SEM_NULL)
		{
			return( RC_SET( FERR_MEM));
		}

		return( FERR_OK);
	}

	FINLINE void f_semDestroy(
		F_SEM *		phSem)
	{
		if (*phSem != F_SEM_NULL)
		{
			(void)kSemaphoreFree( (SEMAPHORE)(*phSem));
			*phSem = F_SEM_NULL;
		}
	}

	FINLINE RCODE f_semWait(
		F_SEM			hSem,
		FLMUINT		uiTimeout)
	{
		RCODE			rc = FERR_OK;

		if( uiTimeout == F_SEM_WAITFOREVER)
		{
			if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
			{
				rc = RC_SET( FERR_MUTEX_UNABLE_TO_LOCK);
			}
		}
		else
		{
			if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
			{
				rc = RC_SET( FERR_MUTEX_UNABLE_TO_LOCK);
			}
		}

		return( rc);
	}

	FINLINE void f_semSignal(
		F_SEM			hSem)
	{
		(void)kSemaphoreSignal( (SEMAPHORE)hSem);
	}

#elif defined( FLM_WIN)

	FINLINE RCODE f_semCreate(
		F_SEM *		phSem)
	{
		if( (*phSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL, 
			0, 10000, NULL )) == NULL)
		{
			return( RC_SET( FERR_MUTEX_OPERATION_FAILED));
		}

		return FERR_OK;
	}												

	FINLINE void f_semDestroy(
		F_SEM *		phSem)
	{
		if (*phSem != F_SEM_NULL)
		{
			CloseHandle( *phSem);
			*phSem = F_SEM_NULL;
		}
	}

	FINLINE RCODE f_semWait(
		F_SEM			hSem,
		FLMUINT		uiTimeout)
	{
		if( WaitForSingleObject( hSem, uiTimeout ) == WAIT_OBJECT_0)
		{
			return( FERR_OK);
		}
		else
		{
			return( RC_SET( FERR_MUTEX_UNABLE_TO_LOCK));
		}
	}

	FINLINE void f_semSignal( 
		F_SEM			hSem)
	{
		(void)ReleaseSemaphore( hSem, 1, NULL);
	}

#elif defined( FLM_UNIX)

	void f_semDestroy(
		F_SEM *	phSem);

	RCODE f_semCreate(
		F_SEM *	phSem);

	RCODE f_semWait(
		F_SEM		hSem,
		FLMUINT	uiTimeout);

	FINLINE void f_semSignal( 
		F_SEM			hSem)
	{
#if defined( FLM_AIX) || defined( FLM_OSX)
		(void)sema_signal( (sema_t *)hSem);
#else
		(void)sem_post( (sem_t *)hSem);
#endif
	}

#else
	#error Platform undefined
#endif

/****************************************************************************
								Random Generation Functions
****************************************************************************/

#define MAX_RANDOM		2147483646L

typedef struct
{
	FLMINT32 i32Seed;
} f_randomGenerator;

/*
	Call f_randomSetSeed to initialize your random-number generator.  Then
call f_randomLong, f_randomChoice, or f_randomTruth to access the series of
random values.
  
	Initialize your generator with f_randomSetSeed( &r, SOME_CONSTANT) to get a
reproducible sequence of pseudo-random numbers.  Using different constant
seeds will give you independent sequences.  The constant can be any number
between 1 and MAX_RANDOM, inclusive. 
  
	Call f_randomLong to get a number randomly distributed between 1 and 
MAX_RANDOM.  This is the basic call, but is usually not as convenient as 
the subsequent functions, all of which call f_randomLong and process the 
result into a more useable form.

			1 <= f_randomLong(&r) <= MAX_RANDOM
  
	Call f_randomChoice to get a number uniformly distributed across a
specified range of integer values.
            
			lo <= f_randomChoice(&r, lo, hi) <= hi
        
	Call f_randomTruth(&r, n) to get a boolean value which is true n percent
of the time (0 <= n <= 100).
            	
			0 <= f_randomTrue(&r, n) <= 1
*/

void f_randomize(
	f_randomGenerator  *	pRand);

void f_randomSetSeed(
	f_randomGenerator  *	pRand,
	FLMINT32					i32seed);

FLMINT32	f_randomLong(
	f_randomGenerator  *	pRand);

FLMINT32 f_randomChoice(
	f_randomGenerator  *	pRand,
	FLMINT32 				lo,
	FLMINT32 				hi);

FLMINT f_randomTruth(
	f_randomGenerator  *	pRand,
	FLMINT					iPercentageTrue);

/****************************************************************************
								Time, date, timestamp functions
****************************************************************************/
typedef struct
{
	FLMUINT16   year;
	FLMBYTE		month;
	FLMBYTE		day;
} F_DATE, * F_DATE_p;

typedef struct
{
	FLMBYTE		hour;
	FLMBYTE		minute;
	FLMBYTE		second;
	FLMBYTE		hundredth;
} F_TIME, * F_TIME_p;

typedef struct
{
	FLMUINT16	year;
	FLMBYTE		month;
	FLMBYTE		day;
	FLMBYTE		hour;
	FLMBYTE		minute;
	FLMBYTE		second;
	FLMBYTE		hundredth;
} F_TMSTAMP, * F_TMSTAMP_p;

void f_timeGetSeconds(
	FLMUINT	*		puiSeconds);

void f_timeGetTimeStamp(
	F_TMSTAMP *		pTimeStamp);

FLMINT f_timeGetLocalOffset( void);

void f_timeSecondsToDate(
	FLMUINT			uiSeconds,
	F_TMSTAMP *		pTimeStamp);

void f_timeDateToSeconds(
	F_TMSTAMP *		pTimeStamp,
	FLMUINT *		puiSeconds);

FLMINT f_timeCompareTimeStamps(
	F_TMSTAMP *		pTimeStamp1,
	F_TMSTAMP *		pTimeStamp2,
	FLMUINT			uiFlag);

#if defined( FLM_UNIX)
	unsigned f_timeGetMilliTime();
#endif

/**********************************************************************
Desc: Atomic Increment, Decrement, Exchange
Note:	Some of this code is derived from the Ximian source code contained
		in that Mono project's atomic.h file. 
**********************************************************************/
#ifndef FLM_HAVE_ATOMICS
	#define FLM_HAVE_ATOMICS
#endif

/*******************************************************************
Desc:
*******************************************************************/
#if defined( FLM_GNUC) && defined( __ia64__)
FINLINE FLMINT32 ia64_compare_and_swap(
	volatile int *		piTarget,
	FLMINT32				i32NewVal,
	FLMINT32				i32CompVal)
{
	FLMINT32 			i32Old;

	asm volatile ("mov ar.ccv = %2 ;;\n\t"
				  "cmpxchg4.acq %0 = [%1], %3, ar.ccv\n\t"
				  : "=r" (i32Old) : "r" (piTarget), 
					 "r" (i32CompVal),
					 "r" (i32NewVal));

	return( i32Old);
}
#endif

/**********************************************************************
Desc:
**********************************************************************/
#if defined( FLM_SPARC) && defined( FLM_SOLARIS) && !defined( FLM_GNUC)
extern "C" FLMINT32 sparc_atomic_add_32(
	volatile FLMINT32 *		piTarget,
	FLMINT32						iDelta);
#endif

/**********************************************************************
Desc:
**********************************************************************/
#if defined( FLM_SPARC) && defined( FLM_SOLARIS) && !defined( FLM_GNUC)
extern "C" FLMINT32 sparc_atomic_xchg_32(
	volatile FLMINT32 *		piTarget,
	FLMINT32						iNewValue);
#endif

/**********************************************************************
Desc:
**********************************************************************/
#if defined( FLM_AIX)
FINLINE int aix_atomic_add(
	volatile int *			piTarget,
	int 						iDelta)
{
	return( fetch_and_add( (int *)piTarget, iDelta) + iDelta);
}
#endif

/**********************************************************************
Desc:
**********************************************************************/
FINLINE FLMINT32 _flmAtomicInc(
	FLMATOMIC *			piTarget)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)nlm_atomic_inc( piTarget);
	}
	#elif defined( FLM_WIN)
	{
		return( (FLMINT32)InterlockedIncrement( (volatile LONG *)piTarget));
	}
	#elif defined( FLM_AIX)
	{
		return( (FLMINT32)aix_atomic_add( piTarget, 1));
	}
	#elif defined( FLM_GNUC)
	{
		#if defined( __i386__) || defined( __x86_64__)
		{
			FLMINT32 			i32Tmp;
			
			__asm__ __volatile__ ("lock; xaddl %0, %1"
							: "=r" (i32Tmp), "=m" (*piTarget)
							: "0" (1), "m" (*piTarget));
		
			return( i32Tmp + 1);
		}
		#elif defined( __ppc__) || defined ( __powerpc__)
		{
			FLMINT32				i32Result = 0;
			FLMINT32				i32Tmp;
		
			__asm__ __volatile__ ("\n1:\n\t"
							"lwarx  %0, 0, %2\n\t"
							"addi   %1, %0, 1\n\t"
							"stwcx. %1, 0, %2\n\t"
							"bne-   1b"
							: "=&b" (i32Result), "=&b" (i32Tmp) 
							: "r" (piTarget) : "cc", "memory");
	
			return( i32Result + 1);
		}
		#elif defined( __ia64__)
		{
			FLMINT32				i32Old;

			for( ;;)
			{
				i32Old = (FLMINT32)*piTarget;
				
				if( ia64_compare_and_swap( piTarget, 
						i32Old + 1, i32Old) == i32Old)
				{
					break;
				}
			}
		
			return( i32Old + 1);
		}
		#elif defined( __s390__)
		{
			FLMINT32				i32Tmp;
			
			__asm__ __volatile__ ("\tLA\t2,%1\n"
							"0:\tL\t%0,%1\n"
							"\tLR\t1,%0\n"
							"\tAHI\t1,1\n"
							"\tCS\t%0,1,0(2)\n"
							"\tJNZ\t0b\n"
							"\tLR\t%0,1"
							: "=r" (i32Tmp), "+m" (*piTarget)
							: : "1", "2", "cc");
		
			return( i32Tmp);
		}
		#else
			#ifdef FLM_HAVE_ATOMICS
				#undef FLM_HAVE_ATOMICS
			#endif

			F_UNREFERENCED_PARM( piTarget);	

			flmAssert( 0);
			return( 0);
		#endif
	}
	#elif defined( FLM_SOLARIS) && defined( FLM_SPARC)
		return( sparc_atomic_add_32( piTarget, 1));
	#else
		#ifdef FLM_HAVE_ATOMICS
			#undef FLM_HAVE_ATOMICS
		#endif

		F_UNREFERENCED_PARM( piTarget);	

		flmAssert( 0);
		return( 0);
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE FLMINT32 _flmAtomicDec(
	FLMATOMIC *			piTarget)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)nlm_atomic_dec( piTarget));
	}
	#elif defined( FLM_WIN)
	{
		return( (FLMINT32)InterlockedDecrement( (volatile LONG *)piTarget));
	}
	#elif defined( FLM_AIX)
	{
		return( (FLMINT32)aix_atomic_add( piTarget, -1));
	}
	#elif defined( FLM_GNUC)
	{
		#if defined( __i386__) || defined( __x86_64__)
		{
			FLMINT32				i32Tmp;
			
			__asm__ __volatile__ ("lock; xaddl %0, %1"
							: "=r" (i32Tmp), "=m" (*piTarget)
							: "0" (-1), "m" (*piTarget));
		
			return( i32Tmp - 1);
		}
		#elif defined( __ppc__) || defined ( __powerpc__)
		{
			FLMINT32				i32Result = 0;
			FLMINT32				i32Tmp;
		
			__asm__ __volatile__ ("\n1:\n\t"
							"lwarx  %0, 0, %2\n\t"
							"addi   %1, %0, -1\n\t"
							"stwcx. %1, 0, %2\n\t"
							"bne-   1b"
							: "=&b" (i32Result), "=&b" (i32Tmp) 
							: "r" (piTarget) : "cc", "memory");
							
			return( i32Result - 1);
		}
		#elif defined( __ia64__)
		{
			FLMINT32				i32Old;

			for( ;;)
			{
				i32Old = (FLMINT32)*piTarget;
				
				if( ia64_compare_and_swap( piTarget, i32Old - 1,
					i32Old) == i32Old)
				{
					break;
				}
			}
		
			return( i32Old - 1);
		}
		#elif defined( __s390__)
		{
			FLMINT32				i32Tmp;
	
			__asm__ __volatile__ ("\tLA\t2,%1\n"
							"0:\tL\t%0,%1\n"
							"\tLR\t1,%0\n"
							"\tAHI\t1,-1\n"
							"\tCS\t%0,1,0(2)\n"
							"\tJNZ\t0b\n"
							"\tLR\t%0,1"
							: "=r" (i32Tmp), "+m" (*piTarget)
							: : "1", "2", "cc");
		
			return( i32Tmp);
		}
		#else
			#ifdef FLM_HAVE_ATOMICS
				#undef FLM_HAVE_ATOMICS
			#endif

			F_UNREFERENCED_PARM( piTarget);
			
			flmAssert( 0);
			return( 0);
		#endif
	}
	#elif defined( FLM_SOLARIS) && defined( FLM_SPARC)
		return( sparc_atomic_add_32( piTarget, -1));
	#else
		#ifdef FLM_HAVE_ATOMICS
			#undef FLM_HAVE_ATOMICS
		#endif

		F_UNREFERENCED_PARM( piTarget);
			
		flmAssert( 0);
		return( 0);
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE FLMINT32 _flmAtomicExchange(
	FLMATOMIC *			piTarget,
	FLMINT32				i32NewVal)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)atomic_xchg( piTarget, i32NewVal));
	}
	#elif defined( FLM_WIN)
	{
		return( (FLMINT32)InterlockedExchange( (volatile LONG *)piTarget,
			i32NewVal));
	}
	#elif defined( FLM_AIX)
	{
		int		iOldVal;
		
		for( ;;)
		{ 
			iOldVal = (int)*piTarget;
			
			if( compare_and_swap( (int *)piTarget, &iOldVal, i32NewVal))
			{
				break;
			}
		}
		
		return( (FLMINT32)iOldVal);
	}
	#elif defined( FLM_GNUC)
	{
		#if defined( __i386__) || defined( __x86_64__)
		{
			FLMINT32 			i32Ret;
			
			__asm__ __volatile__ ("1:; lock; cmpxchgl %2, %0; jne 1b"
							: "=m" (*piTarget), "=a" (i32Ret)
							: "r" (i32NewVal), "m" (*piTarget),
							  "a" (*piTarget));
		
			return( i32Ret);
		}
		#elif defined( __ppc__) || defined ( __powerpc__)
		{
			FLMINT32				i32Tmp = 0;
		
			__asm__ __volatile__ ("\n1:\n\t"
							"lwarx  %0, 0, %2\n\t"
							"stwcx. %3, 0, %2\n\t"
							"bne    1b"
							: "=r" (i32Tmp) : "0" (i32Tmp), 
							  "b" (piTarget),
							  "r" (i32NewVal) : "cc", "memory");
							
			return( i32Tmp);
		}
		#elif defined( __ia64__)
		{
			FLMINT32			i32Result;

			for( ;;)
			{
				i32Result = (FLMINT32)*piTarget;
				
				if( ia64_compare_and_swap( piTarget, 
						i32NewVal, i32Result) == i32Result)
				{
					break;
				}
			}
		
			return( i32Result);
		}
		#elif defined( __s390__)
		{
			FLMINT32				i32Ret;
			
			__asm__ __volatile__ ("\tLA\t1,%0\n"
							"0:\tL\t%1,%0\n"
							"\tCS\t%1,%2,0(1)\n"
							"\tJNZ\t0b"
							: "+m" (*piTarget), "=r" (i32Ret)
							: "r" (i32NewVal)
							: "1", "cc");
		
			return( i32Ret);
		}
		#else
			#ifdef FLM_HAVE_ATOMICS
				#undef FLM_HAVE_ATOMICS
			#endif

			F_UNREFERENCED_PARM( piTarget);
			F_UNREFERENCED_PARM( i32NewVal);
			
			flmAssert( 0);
			return( 0);
		#endif
	}
	#elif defined( FLM_SOLARIS) && defined( FLM_SPARC)
		return( sparc_atomic_xchg_32( piTarget, i32NewVal));
	#else
		#ifdef FLM_HAVE_ATOMICS
			#undef FLM_HAVE_ATOMICS
		#endif

		F_UNREFERENCED_PARM( piTarget);
		F_UNREFERENCED_PARM( i32NewVal);

		flmAssert( 0);
		return( 0);
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE FLMINT32 flmAtomicInc(
	FLMATOMIC *		piTarget,
	F_MUTEX			hMutex = F_MUTEX_NULL,
	FLMBOOL			bMutexAlreadyLocked = FALSE)
{
	#ifdef FLM_HAVE_ATOMICS
		F_UNREFERENCED_PARM( bMutexAlreadyLocked);
		F_UNREFERENCED_PARM( hMutex);
		
		return( _flmAtomicInc( piTarget));
	#else
	{
		FLMINT32		i32NewVal;
		
		flmAssert( hMutex != F_MUTEX_NULL);

		if( !bMutexAlreadyLocked)
		{
			f_mutexLock( hMutex);
		}
		
		i32NewVal = (FLMINT32)(++(*piTarget));
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( hMutex);
		}
		
		return( i32NewVal);
	}
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE FLMINT32 flmAtomicDec(
	FLMATOMIC *		piTarget,
	F_MUTEX			hMutex = F_MUTEX_NULL,
	FLMBOOL			bMutexAlreadyLocked = FALSE)
{
	#ifdef FLM_HAVE_ATOMICS
		F_UNREFERENCED_PARM( bMutexAlreadyLocked);
		F_UNREFERENCED_PARM( hMutex);
		
		return( _flmAtomicDec( piTarget));
	#else
	{
		FLMINT32		i32NewVal;
		
		flmAssert( hMutex != F_MUTEX_NULL);
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexLock( hMutex);
		}
		
		i32NewVal = (FLMINT32)(--(*piTarget));
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( hMutex);
		}
		
		return( i32NewVal);
	}
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FINLINE FLMINT32 flmAtomicExchange(
	FLMATOMIC *		piTarget,
	FLMINT32			i32NewVal,
	F_MUTEX			hMutex = F_MUTEX_NULL,
	FLMBOOL			bMutexAlreadyLocked = FALSE)
{
	#ifdef FLM_HAVE_ATOMICS
		F_UNREFERENCED_PARM( bMutexAlreadyLocked);
		F_UNREFERENCED_PARM( hMutex);
		
		return( _flmAtomicExchange( piTarget, i32NewVal));
	#else
	{
		FLMINT32		i32OldVal;
		
		flmAssert( hMutex != F_MUTEX_NULL);
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexLock( hMutex);
		}
		
		i32OldVal = (FLMINT32)*piTarget;
		*piTarget = i32NewVal;
		
		if( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( hMutex);
		}
		
		return( i32OldVal);
	}
	#endif
}

/****************************************************************************
									Pseudo Serial Numbers
****************************************************************************/

RCODE f_initSerialNumberGenerator( void);

RCODE f_createSerialNumber(
	FLMBYTE *		pszGuid);

void f_freeSerialNumberGenerator( void);

/****************************************************************************
									       CRC
****************************************************************************/

RCODE f_initCRCTable(
	FLMUINT32 **	ppui32CRCTbl);

void f_updateCRC(
	FLMUINT32 *		pui32CRCTbl,
	FLMBYTE *		pucBlk,
	FLMUINT			uiBlkSize,
	FLMUINT32 *		pui32CRC);

#define f_freeCRCTable( ppui32CRCTbl) \
	f_free( ppui32CRCTbl)

/****************************************************************************
Desc: Returns TRUE if the passed-in character is 0-9, a-f, or A-F
****************************************************************************/
FINLINE FLMBOOL f_isHexChar(
	FLMBYTE		ucChar)
{
	if( (ucChar >= '0' && ucChar <= '9') ||
		(ucChar >= 'A' && ucChar <= 'F') ||
		(ucChar >= 'a' && ucChar <= 'f'))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc: Returns the base-10 equivalent of a hex character
****************************************************************************/
FINLINE FLMBYTE f_getHexVal(
	FLMBYTE		ucChar)
{
	if( ucChar >= '0' && ucChar <= '9')
	{
		return( (FLMBYTE)(ucChar - '0'));
	}
	else if( ucChar >= 'A' && ucChar <= 'F')
	{
		return( (FLMBYTE)((ucChar - 'A') + 10));
	}
	else if( ucChar >= 'a' && ucChar <= 'f')
	{
		return( (FLMBYTE)((ucChar - 'a') + 10));
	}

	return( 0);
}

/****************************************************************************
									Process ID Functions
	****************************************************************************/

#if defined( FLM_WIN)

	FINLINE FLMUINT f_getpid()
	{ 
		return _getpid();
	}

#elif defined( FLM_UNIX)

	pid_t getpid( void);

	FINLINE FLMUINT f_getpid() 
	{ 
		return getpid();
	}

#elif defined( FLM_NLM)

	FLMUINT f_getNLMHandle( void);

	FINLINE FLMUINT f_getpid() 
	{ 
		return( f_getNLMHandle());
	}

#endif
																		
typedef struct
{
	char *	pszDestStr;
} F_SPRINTF_INFO;

void flmSprintfStringFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

FLMINT f_vsprintf(
	char *				pszDestStr,
	const char *		pszFormat,
	f_va_list *			args);

typedef FLMINT (* F_SORT_COMPARE_FUNC)(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2);

typedef void (* F_SORT_SWAP_FUNC)(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2);

void f_qsort(
	void *					pvBuffer,
	FLMUINT					uiLowerBounds,
	FLMUINT					uiUpperBounds,
	F_SORT_COMPARE_FUNC	fnCompare,
	F_SORT_SWAP_FUNC		fnSwap);

FLMINT flmQSortUINTCompare(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2);

void flmQSortUINTSwap(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2);
	
/****************************************************************************
									Module Load/Unload Functions
	****************************************************************************/

typedef	void * FlmModHandle;

RCODE FlmModLoad( 
	const char *	 	pszName, 
	FlmModHandle *		phMod);

#ifndef FLM_NLM
	RCODE FlmModUnload( 
		FlmModHandle * 	phMod);
#else
	RCODE FlmModUnload( 
		const char *	pszModPath);
#endif

RCODE FlmSymLoad(
	const char *		pszName, 
	FlmModHandle 		hMod,  
	void ** 				ppvSym);

RCODE FlmSymUnload(
  const char *			pszName);

char * f_strchr(
	const char *	pszStr,
	char				c);

#include "fpackoff.h"
#endif
