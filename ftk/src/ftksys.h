//------------------------------------------------------------------------------
// Desc:	Cross-platform macros, defines, etc.  Must visit this file
//			to port XFLAIM to another platform.
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
// $Id: ftk.h 3123 2006-01-24 17:19:50 -0700 (Tue, 24 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#ifndef FTKSYS_H
#define FTKSYS_H

	#include "ftk.h"
	
	#ifdef FLM_NLM
		#if !defined( FLM_RING_ZERO_NLM) && !defined( FLM_LIBC_NLM)
			#define FLM_LIBC_NLM
		#endif
	
		#if defined( FLM_RING_ZERO_NLM) && defined( FLM_LIBC_NLM)
			#error Cannot target both LIBC and RING 0
		#endif
	#endif

	class F_FileHdl;
	class F_Thread;
	class F_ThreadMgr;
	class F_IOBufferMgr;
	class F_FileSystem;
	class F_ThreadMgr;
	class F_ResultSet;
	class F_ResultSetBlk;

	/****************************************************************************
	Desc: Global data
	****************************************************************************/

	#define FLM_DEFAULT_OPEN_THRESHOLD						100
	#define FLM_DEFAULT_MAX_AVAIL_TIME						900
	#define FLM_MAX_KEY_SIZE									1024
	#define FLM_NLM_SECTOR_SIZE								512

	/****************************************************************************
	Desc:		NLM
	****************************************************************************/
	#if defined( FLM_NLM)
		#include "ftknlm.h"
	#endif

	/****************************************************************************
	Desc:	WIN
	****************************************************************************/
	#if defined( FLM_WIN)

		#ifndef WIN32_LEAN_AND_MEAN
			#define WIN32_LEAN_AND_MEAN
		#endif
	
		#ifndef WIN32_EXTRA_LEAN
			#define WIN32_EXTRA_LEAN
		#endif
	
		// Enable critical section and spin count API to be visible in header
		// file.
	
		#define _WIN32_WINNT	0x0403
	
		#pragma pack( push, enter_windows, 8)
			#include <windows.h>
			#include <time.h>
			#include <stdlib.h>
			#include <stddef.h>
			#include <rpc.h>
			#include <process.h>
			#include <winsock.h>
			#include <imagehlp.h>
			#include <malloc.h>
			#include <stdio.h>
			#include <direct.h>
		#pragma pack( pop, enter_windows)
		
		// Conversion from XXX to YYY, possible loss of data
		#pragma warning( disable : 4244) 
	
		// Local variable XXX may be used without having been initialized
		#pragma warning( disable : 4701)
	
		// Function XXX not inlined
		#pragma warning( disable : 4710) 
		
		#define ENDLINE			ENDLINE_CRLF
		
	#endif

	/****************************************************************************
	Desc:		UNIX
	****************************************************************************/
	#if defined( FLM_UNIX)

		#ifdef HAVE_CONFIG_H
			#include "config.h"
		#endif

		#ifdef FLM_AIX
			#ifndef _LARGE_FILES
				#define _LARGE_FILES
			#endif
			#include <sys/vminfo.h>
		#endif

		#include <stdio.h>
		#include <fcntl.h>
		#include <assert.h>
		#include <pthread.h>
		#include <errno.h>
		#include <glob.h>
		#include <limits.h>
		#include <netdb.h>
		#include <sys/types.h>
		#include <netinet/in.h>
		#include <arpa/nameser.h>
		#include <resolv.h>
		#include <stdarg.h>
		#include <stdlib.h>
		#include <string.h>
		#include <strings.h>
		#include <time.h>
		#include <unistd.h>
		#include <utime.h>
		#include <aio.h>
		#include <arpa/inet.h>
		#include <netinet/tcp.h>
		#include <sys/mman.h>
		#include <sys/resource.h>
		#include <sys/socket.h>
		#include <sys/stat.h>
		#include <sys/time.h>

		#ifdef FLM_AIX
			#include <sys/atomic_op.h>
		#endif

		#ifdef FLM_OSX
			#include <sys/resource.h>
			#include <sys/param.h>
			#include <sys/mount.h>
			#include <libkern/OSAtomic.h>
		#endif

		#ifdef FLM_SOLARIS
			#include <signal.h>
			#include <synch.h>
		#endif

		typedef int						SOCKET;
		#define INVALID_SOCKET		-1
	
	#endif

	#if defined( __va_copy)
		#define  f_va_copy(to, from) __va_copy(to, from)
	#else
		#define f_va_copy(to, from)  ((to) = (from))
	#endif

	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_SOLARIS) && defined( FLM_SPARC_PLUS) && !defined( FLM_GNUC)
	extern "C" FLMINT32 sparc_atomic_add_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_SOLARIS) && defined( FLM_SPARC_PLUS) && !defined( FLM_GNUC)
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
	#if defined( FLM_UNIX)
	FLMINT32 posix_atomic_add_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_UNIX)
	FLMINT32 posix_atomic_xchg_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iNewValue);
	#endif
	
	/****************************************************************************
	Desc: Mutex and semaphore routines
	****************************************************************************/
	#if defined( FLM_WIN)
		typedef struct
		{
			FLMATOMIC						locked;
	#ifdef FLM_DEBUG
			FLMUINT							uiThreadId;
			FLMATOMIC						lockedCount;
			FLMATOMIC						waitCount;
	#endif
		} F_INTERLOCK;
	#endif

	/****************************************************************************
											 f_sprintf
	****************************************************************************/

	// Percent formating prefixes
	
	#define FLM_PREFIX_NONE				0
	#define FLM_PREFIX_MINUS 			1
	#define FLM_PREFIX_PLUS				2
	#define FLM_PREFIX_POUND 			3
	
	// Width and Precision flags
	
	#define FLM_PRINTF_MINUS_FLAG		0x0001
	#define FLM_PRINTF_PLUS_FLAG		0x0002
	#define FLM_PRINTF_SPACE_FLAG		0x0004
	#define FLM_PRINTF_POUND_FLAG		0x0008
	#define FLM_PRINTF_ZERO_FLAG		0x0010
	#define FLM_PRINTF_SHORT_FLAG		0x0020
	#define FLM_PRINTF_LONG_FLAG		0x0040
	#define FLM_PRINTF_DOUBLE_FLAG	0x0080
	#define FLM_PRINTF_INT64_FLAG		0x0100
	#define FLM_PRINTF_COMMA_FLAG		0x0200

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IOBuffer : public IF_IOBuffer
	{
	public:
	
		F_IOBuffer();
	
		virtual ~F_IOBuffer();
	
		RCODE FLMAPI setupBuffer(
			FLMUINT	uiBufferSize,
			FLMUINT	uiBlockSize);
	
		FINLINE FLMBYTE * FLMAPI getBuffer( void)
		{
			return( m_pucBuffer);
		}
	
		FINLINE FLMUINT FLMAPI getBufferSize( void)
		{
			return( m_uiBufferSize);
		}
	
		FINLINE FLMUINT FLMAPI getBlockSize( void)
		{
			return( m_uiBlockSize);
		}
	
		void FLMAPI notifyComplete(
			RCODE			rc);
			
		void FLMAPI signalComplete(
			RCODE			rc);
	
		FINLINE void FLMAPI setCompletionCallback(
			WRITE_COMPLETION_CB 	fnCompletion)
		{
			m_fnCompletion = fnCompletion;
		}
	
		FINLINE void FLMAPI setCompletionCallbackData(
			FLMUINT	uiBlockNumber,
			void *	pvData)
		{
			f_assert( uiBlockNumber < FLM_MAX_IO_BUFFER_BLOCKS);
			m_UserData [uiBlockNumber] = pvData;
		}
	
		FINLINE void * FLMAPI getCompletionCallbackData(
			FLMUINT	uiBlockNumber)
		{
			f_assert( uiBlockNumber < FLM_MAX_IO_BUFFER_BLOCKS);
			return( m_UserData [uiBlockNumber]);
		}
	
		FINLINE RCODE FLMAPI getCompletionCode( void)
		{
			return( m_completionRc);
		}
	
		FINLINE eBufferMgrList FLMAPI getList( void)
		{
			return( m_eList);
		}
	
		void FLMAPI makePending( void);
		
		FLMBOOL FLMAPI isPending( void);
	
		void FLMAPI startTimer(
			void *					pvStats);
			
		void * FLMAPI getStats( void);
		
		FLMUINT64 FLMAPI getElapTime( void);
		
	#ifdef FLM_WIN
		FINLINE OVERLAPPED * getOverlapped( void)
		{
			return( &m_Overlapped);
		}
	
		FINLINE void setFileHandle(
			HANDLE	FileHandle)
		{
			m_FileHandle = FileHandle;
		}
	#endif
	
	#if defined( FLM_LINUX) || defined( FLM_SOLARIS) || defined( FLM_OSX)
		FINLINE struct aiocb * getAIOStruct( void)
		{
			return( &m_aio);
		}
	#endif
	
	private:
	
		// Only called by the buffer manager
	
		RCODE setupIOBuffer(
			IF_IOBufferMgr *	pIOBufferMgr);
	
		FLMBOOL isIOComplete( void);
	
		RCODE waitToComplete( void);
	
		// Private methods and variables
	
		F_IOBufferMgr *		m_pIOBufferMgr;
		FLMBYTE *				m_pucBuffer;
		void *					m_UserData[ FLM_MAX_IO_BUFFER_BLOCKS];
		FLMUINT					m_uiBufferSize;
		FLMUINT					m_uiBlockSize;
		eBufferMgrList			m_eList;
		FLMBOOL					m_bDeleteOnNotify;
	#ifdef FLM_WIN
		HANDLE					m_FileHandle;
		OVERLAPPED				m_Overlapped;
	#endif
	#if defined( FLM_LINUX) || defined( FLM_SOLARIS) || defined( FLM_OSX)
		struct aiocb			m_aio;
	#endif
		F_IOBuffer *			m_pNext;
		F_IOBuffer *			m_pPrev;
		WRITE_COMPLETION_CB	m_fnCompletion;
		RCODE						m_completionRc;
		F_TMSTAMP				m_StartTime;
		FLMUINT64				m_ui64ElapMilli;
		void *					m_pStats;
#ifdef FLM_RING_ZERO_NLM
		SEMAPHORE				m_hSem;
#endif
	
		friend class F_IOBufferMgr;
	};

	/***************************************************************************
	Desc:
	***************************************************************************/
	#ifdef FLM_WIN
	class F_FileHdl : public IF_FileHdl
	{
	public:
	
		F_FileHdl();
	
		virtual ~F_FileHdl();
	
		RCODE FLMAPI close( void);
		
		RCODE FLMAPI flush( void);
	
		RCODE FLMAPI read(
			FLMUINT64		ui64Offset,
			FLMUINT			uiLength,
			void *			pvBuffer,
			FLMUINT *		puiBytesRead);
	
		RCODE FLMAPI seek(
			FLMUINT64		ui64Offset,
			FLMINT			iWhence,
			FLMUINT64 *		pui64NewOffset);
	
		RCODE FLMAPI size(
			FLMUINT64 *		pui64Size);
	
		RCODE FLMAPI tell(
			FLMUINT64 *		pui64Offset);
	
		RCODE FLMAPI truncate(
			FLMUINT64		ui64Size);
	
		RCODE FLMAPI write(
			FLMUINT64		ui64Offset,
			FLMUINT			uiLength,
			const void *	pvBuffer,
			FLMUINT *		puiBytesWritten);
	
		FINLINE RCODE FLMAPI sectorRead(
			FLMUINT64		ui64ReadOffset,
			FLMUINT			uiBytesToRead,
			void *			pvBuffer,
			FLMUINT *		puiBytesReadRV)
		{
			if (m_bDoDirectIO)
			{
				return( directRead( ui64ReadOffset, uiBytesToRead,
						pvBuffer, puiBytesReadRV));
			}
			else
			{
				return( read( ui64ReadOffset, uiBytesToRead, 
						pvBuffer, puiBytesReadRV));
			}
		}
	
		FINLINE RCODE FLMAPI sectorWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			IF_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV)
		{
			if (m_bDoDirectIO)
			{
				return( directWrite( ui64WriteOffset, uiBytesToWrite,
						pvBuffer, pBufferObj, puiBytesWrittenRV));
			}
			else
			{
				f_assert( !pBufferObj);
				return( write( ui64WriteOffset, uiBytesToWrite, pvBuffer,
									puiBytesWrittenRV));
			}
		}
	
		FINLINE FLMBOOL FLMAPI canDoAsync( void)
		{
			return( m_bOpenedInAsyncMode);
		}
	
		FINLINE void FLMAPI setExtendSize(
			FLMUINT		uiExtendSize)
		{
			m_uiExtendSize = uiExtendSize;
		}
	
		FINLINE void FLMAPI setMaxAutoExtendSize(
			FLMUINT		uiMaxAutoExtendSize)
		{
			m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
		}
	
		FINLINE FLMBOOL FLMAPI isReadOnly( void)
		{
			return( m_bOpenedReadOnly);
		}
		
		RCODE FLMAPI lock( void);
	
		RCODE FLMAPI unlock( void);
		
		FINLINE FLMUINT FLMAPI getSectorSize( void)
		{
			return( m_uiBytesPerSector);
		}
		
	private:
	
		RCODE create(
			const char *	pszFileName,
			FLMUINT			uiIoFlags);
	
		RCODE createUnique(
			char *			pszDirName,
			const char *	pszFileExtension,
			FLMUINT			uiIoFlags);
	
		RCODE open(
			const char *	pszFileName,
			FLMUINT			uiIoFlags);
	
		FINLINE HANDLE getFileHandle( void)
		{
			return m_FileHandle;
		}
	
		RCODE openOrCreate(
			const char *	pszFileName,
			FLMUINT			uiAccess,
			FLMBOOL			bCreateFlag);
	
		RCODE doOneRead(
			FLMUINT64		ui64Offset,
			FLMUINT			uiLength,
			void *			pvBuffer,
			FLMUINT *		puiBytesRead);
	
		RCODE directRead(
			FLMUINT64		uiOffset,
			FLMUINT			uiLength,
			void *			pvBuffer,
			FLMUINT *		puiBytesRead);
	
		RCODE directWrite(
			FLMUINT64		uiOffset,
			FLMUINT			uiLength,
			const void *	pvBuffer,
			IF_IOBuffer *	pBufferObj,
			FLMUINT *		puiBytesWritten);
	
		FINLINE FLMUINT64 roundToNextSector(
			FLMUINT64		ui64Bytes)
		{
			return( (ui64Bytes + m_ui64NotOnSectorBoundMask) & 
							m_ui64GetSectorBoundMask);
		}
	
		FINLINE FLMUINT64 truncateToPrevSector(
			FLMUINT64		ui64Offset)
		{
			return( ui64Offset & m_ui64GetSectorBoundMask);
		}
	
		RCODE extendFile(
			FLMUINT64		ui64EndOfLastWrite,
			FLMUINT			uiMaxBytesToExtend,
			FLMBOOL			bFlush);
			
		RCODE allocAlignedBuffer( void);
	
		FLMBOOL				m_bFileOpened;
		FLMBOOL				m_bDeleteOnRelease;
		FLMBOOL				m_bOpenedReadOnly;
		FLMBOOL				m_bOpenedExclusive;
		char *				m_pszFileName;
		HANDLE				m_FileHandle;
		FLMUINT				m_uiBytesPerSector;
		FLMUINT64			m_ui64NotOnSectorBoundMask;
		FLMUINT64			m_ui64GetSectorBoundMask;
		FLMBOOL				m_bDoDirectIO;
		FLMUINT				m_uiExtendSize;
		FLMUINT				m_uiMaxAutoExtendSize;
		FLMBYTE *			m_pucAlignedBuff;
		FLMUINT				m_uiAlignedBuffSize;
		FLMUINT64			m_ui64CurrentPos;
		FLMBOOL				m_bOpenedInAsyncMode;
		OVERLAPPED			m_Overlapped;
		
		friend class F_FileSystem;
		friend class F_MultiFileHdl;
	};
	#endif

	/***************************************************************************
	Desc:
	***************************************************************************/
	#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	class F_FileHdl : public IF_FileHdl
	{
	public:
	
		F_FileHdl();
	
		~F_FileHdl();
	
		RCODE FLMAPI flush( void);
	
		RCODE FLMAPI read(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
	
		RCODE FLMAPI seek(
			FLMUINT64			ui64Offset,
			FLMINT				iWhence,
			FLMUINT64 *			pui64NewOffset);
	
		RCODE FLMAPI size(
			FLMUINT64 *			pui64Size);
	
		RCODE FLMAPI tell(
			FLMUINT64 *			pui64Offset);
	
		RCODE FLMAPI truncate(
			FLMUINT64			ui64Size);
	
		RCODE FLMAPI write(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			const void *		pvBuffer,
			FLMUINT *			puiBytesWritten);
	
		RCODE FLMAPI sectorRead(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,
			void *				pvBuffer,
			FLMUINT *			puiBytesReadRV);
	
		RCODE FLMAPI sectorWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			IF_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV)
		{
			if( m_bDoDirectIO)
			{
				return( directWrite( ui64WriteOffset, uiBytesToWrite, 
					pvBuffer, pBufferObj, puiBytesWrittenRV));
			}
			else
			{
				return( write( ui64WriteOffset, uiBytesToWrite,
					pvBuffer, puiBytesWrittenRV));
			}
		}
	
		RCODE FLMAPI close( void);
													
		FLMBOOL FLMAPI canDoAsync( void);
	
		FINLINE FLMBOOL FLMAPI usingDirectIo( void)
		{
			return( m_bDoDirectIO);
		}
	
		FINLINE void FLMAPI setExtendSize(
			FLMUINT				uiExtendSize)
		{
			m_uiExtendSize = uiExtendSize;
		}
	
		FINLINE void FLMAPI setMaxAutoExtendSize(
			FLMUINT)
		{
		}
	
		RCODE FLMAPI lock( void);
	
		RCODE FLMAPI unlock( void);
	
		FINLINE FLMBOOL FLMAPI isReadOnly( void)
		{
			return( m_bOpenedReadOnly);
		}
		
		FINLINE FLMUINT FLMAPI getSectorSize( void)
		{
			return( m_uiBytesPerSector);
		}
	
	private:
	
		RCODE create(
			const char *		pszFileName,
			FLMUINT				uiIoFlags);
	
		RCODE createUnique(
			char *				pszDirName,
			const char *		pszFileExtension,
			FLMUINT				uiIoFlags);
	
		RCODE open(
			const char *		pszFileName,
			FLMUINT				uiIoFlags);
	
		RCODE openOrCreate(
			const char *		pszFileName,
			FLMUINT				uiAccess,
			FLMBOOL				bCreateFlag);
	
		FINLINE FLMUINT64 roundToNextSector(
			FLMUINT64		ui64Bytes)
		{
			return( (ui64Bytes + m_ui64NotOnSectorBoundMask) & 
							m_ui64GetSectorBoundMask);
		}
		
		FINLINE FLMUINT64 truncateToPrevSector(
			FLMUINT64		ui64Offset)
		{
			return( ui64Offset & m_ui64GetSectorBoundMask);
		}
	
		RCODE doOneRead(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
			
		RCODE directRead(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,	
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
	
		RCODE directWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			IF_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV);
		
		RCODE allocAlignedBuffer( void);
		
		FLMBOOL					m_bFileOpened;
		FLMBOOL					m_bDeleteOnRelease;
		FLMBOOL					m_bOpenedReadOnly;
		FLMBOOL					m_bOpenedExclusive;
		char *					m_pszFileName;
		int				   	m_fd;
		FLMUINT					m_uiBytesPerSector;
		FLMUINT64				m_ui64NotOnSectorBoundMask;
		FLMUINT64				m_ui64GetSectorBoundMask;
		FLMUINT64				m_ui64CurrentPos;
		FLMUINT					m_uiExtendSize;
		FLMBOOL					m_bOpenedInAsyncMode;
		FLMBOOL					m_bDoDirectIO;
		FLMBYTE *				m_pucAlignedBuff;
		FLMUINT					m_uiAlignedBuffSize;
		
		friend class F_FileSystem;
		friend class F_MultiFileHdl;
	};
	#endif
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	#if defined( FLM_RING_ZERO_NLM)
	class F_FileHdl : public IF_FileHdl
	{
	public:
	
		F_FileHdl();
	
		virtual ~F_FileHdl();
	
		RCODE FLMAPI flush( void);
	
		RCODE FLMAPI read(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
	
		RCODE FLMAPI seek(
			FLMUINT64			ui64Offset,
			FLMINT				iWhence,
			FLMUINT64 *			pui64NewOffset);
	
		RCODE FLMAPI size(
			FLMUINT64 *			pui64Size);
	
		RCODE FLMAPI tell(
			FLMUINT64 *			pui64Offset);
	
		RCODE FLMAPI truncate(
			FLMUINT64			ui64Size);
	
		RCODE FLMAPI write(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			const void *		pvBuffer,
			FLMUINT *			puiBytesWritten);
	
		RCODE FLMAPI sectorRead(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,
			void *				pvBuffer,
			FLMUINT *			puiBytesReadRV);
	
		RCODE FLMAPI sectorWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			IF_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV);
	
		RCODE FLMAPI close( void);
													
		FLMBOOL FLMAPI canDoAsync( void);
	
		FINLINE void FLMAPI setExtendSize(
			FLMUINT				uiExtendSize)
		{
			m_uiExtendSize = uiExtendSize;
		}
	
		FINLINE void FLMAPI setMaxAutoExtendSize(
			FLMUINT				uiMaxAutoExtendSize)
		{
			m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
		}
	
		FINLINE void FLMAPI setSuballocation(
			FLMBOOL				bDoSuballocation)
		{
			m_bDoSuballocation = bDoSuballocation;
		}
	
		FINLINE FLMUINT FLMAPI getSectorSize( void)
		{
			return( FLM_NLM_SECTOR_SIZE);
		}
	
		FINLINE FLMBOOL FLMAPI isReadOnly( void)
		{
			return( m_bOpenedReadOnly);
		}
		
		RCODE FLMAPI lock( void);
	
		RCODE FLMAPI unlock( void);
		
	private:
	
		RCODE setup( void);							
	
		RCODE create(
			const char *		pszFileName,
			FLMUINT				uiIoFlags);
			
		RCODE createUnique(
			char *				pszDirName,
			const char *		pszFileExtension,
			FLMUINT				uiIoFlags);
	
		RCODE open(
			const char *		pszFileName,
			FLMUINT				uiIoFlags);
			
		RCODE openOrCreate(
			const char	*		pszFileName,
			FLMUINT				uiAccess,
			FLMBOOL				bCreateFlag);
	
		RCODE _read(
			FLMUINT				uiOffset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
	
		RCODE _directIORead(
			FLMUINT				uiOffset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
	
		RCODE _directIOSectorRead(
			FLMUINT				uiReadOffset,
			FLMUINT				uiBytesToRead,	
			void *				pvBuffer,
			FLMUINT *			puiBytesReadRV);
	
		RCODE _write(
			FLMUINT				uiWriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			FLMUINT *			puiBytesWrittenRV);
	
		RCODE _directIOWrite(
			FLMUINT				uiWriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			FLMUINT *			puiBytesWrittenRV);
	
		RCODE expand(
			LONG					lStartSector,
			LONG					lSectorsToAlloc);
	
		RCODE writeSectors(
			void *				pvBuffer,
			LONG					lStartSector,
			LONG					lSectorCount,
			IF_IOBuffer *		pBufferObj,
			FLMBOOL *			pbDidAsync = NULL);
	
		RCODE _directIOSectorWrite(
			FLMUINT				uiWriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			IF_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV);
	
		char *					m_pszIoPath;
		FLMBOOL					m_bDeleteOnClose;
		FLMUINT					m_uiMaxFileSize;
		FLMBOOL					m_bFileOpened;
		FLMBOOL					m_bOpenedExclusive;
		FLMBOOL					m_bOpenedReadOnly;
		LONG						m_lFileHandle;
		LONG						m_lOpenAttr;
		LONG						m_lVolumeID;
		LONG						m_lLNamePathCount;
		FLMBOOL					m_bDoSuballocation;
		FLMUINT					m_uiExtendSize;
		FLMUINT					m_uiMaxAutoExtendSize;
		FLMBOOL					m_bDoDirectIO;
		LONG						m_lSectorsPerBlock;
		LONG						m_lMaxBlocks;
		FLMUINT					m_uiCurrentPos;
		FLMBOOL					m_bNSS;
		FLMINT64					m_NssKey;
		FLMBOOL					m_bNSSFileOpen;

		friend class F_FileSystem;
		friend class F_MultiFileHdl;
	};
	#endif

	/***************************************************************************
	Desc:
	***************************************************************************/
	#if 0
	class F_FileHdlMgr : public IF_FileHdlMgr
	{
	public:
	
		F_FileHdlMgr();
	
		virtual FINLINE ~F_FileHdlMgr()
		{
			if (m_hMutex != F_MUTEX_NULL)
			{
				lockMutex( FALSE);
				freeUsedList( TRUE);
				freeAvailList( TRUE);
				unlockMutex( FALSE);
				f_mutexDestroy( &m_hMutex);
			}
		}
	
		FINLINE void FLMAPI setOpenThreshold(
			FLMUINT		uiOpenThreshold)
		{
			if (m_bIsSetup)
			{
				lockMutex( FALSE);
				m_uiOpenThreshold = uiOpenThreshold;
				unlockMutex( FALSE);
			}
		}
	
		FINLINE void FLMAPI setMaxAvailTime(
			FLMUINT		uiMaxAvailTime)
		{
			if (m_bIsSetup)
			{
				lockMutex( FALSE);
				m_uiMaxAvailTime = uiMaxAvailTime;
				unlockMutex( FALSE);
			}
		}
	
		FINLINE FLMUINT FLMAPI getUniqueId( void)
		{
			FLMUINT	uiTemp;
	
			lockMutex( FALSE);
			uiTemp = ++m_uiFileIdCounter;
			unlockMutex( FALSE);
			return( uiTemp);
		}
	
		void FLMAPI findAvail(
			FLMUINT			uiFileId,
			IF_FileHdl **	ppFileHdl);
	
		void FLMAPI removeFileHdls(
			FLMUINT			uiFileId);
	
		void FLMAPI checkAgedFileHdls(
			FLMUINT			uiMinSecondsOpened);
	
		FINLINE FLMUINT FLMAPI getOpenThreshold( void)
		{
			return( m_uiOpenThreshold);
		}
	
		FINLINE FLMUINT FLMAPI getOpenedFiles( void)
		{
			FLMUINT		uiTemp;
	
			lockMutex( FALSE);
			uiTemp = m_uiNumUsed + m_uiNumAvail;
			unlockMutex( FALSE);
			return( uiTemp);
		}
	
		FINLINE FLMUINT FLMAPI getMaxAvailTime( void)
		{
			return( m_uiMaxAvailTime);
		}
	
	private:
	
		RCODE setupFileHdlMgr(
			FLMUINT		uiOpenThreshold = FLM_DEFAULT_OPEN_THRESHOLD,
			FLMUINT		uiMaxAvailTime = FLM_DEFAULT_MAX_AVAIL_TIME);
	
		void freeAvailList(
			FLMBOOL			bMutexAlreadyLocked);
	
		void freeUsedList(
			FLMBOOL			bMutexAlreadyLocked);
	
		FINLINE void insertInUsedList(
			FLMBOOL			bMutexAlreadyLocked,
			IF_FileHdl *	pFileHdl,
			FLMBOOL			bInsertAtEnd)
		{
			insertInList( bMutexAlreadyLocked,
				pFileHdl, bInsertAtEnd,
				&m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);
		}
	
		void makeAvailAndRelease(
			FLMBOOL			bMutexAlreadyLocked,
			IF_FileHdl *	pFileHdl);
	
		void FINLINE releaseOneAvail(
			FLMBOOL			bMutexAlreadyLocked)
		{
			lockMutex( bMutexAlreadyLocked);
			if (m_pFirstAvail)
			{
				removeFromList( TRUE,
					m_pFirstAvail, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
			}
			unlockMutex( bMutexAlreadyLocked);
		}
	
		void insertInList(
			FLMBOOL				bMutexAlreadyLocked,
			IF_FileHdl *		pFileHdl,
			FLMBOOL				bInsertAtEnd,
			IF_FileHdl **		ppFirst,
			IF_FileHdl **		ppLast,
			FLMUINT *			puiCount);
	
		void removeFromList(
			FLMBOOL				bMutexAlreadyLocked,
			IF_FileHdl *		pFileHdl,
			IF_FileHdl **		ppFirst,
			IF_FileHdl **		ppLast,
			FLMUINT *			puiCount);
	
		FINLINE void lockMutex(
			FLMBOOL				bMutexAlreadyLocked)
		{
			if (m_hMutex != F_MUTEX_NULL && !bMutexAlreadyLocked)
			{
				f_mutexLock( m_hMutex);
			}
		}
	
		FINLINE void unlockMutex(
			FLMBOOL				bMutexAlreadyLocked)
		{
			if (m_hMutex != F_MUTEX_NULL && !bMutexAlreadyLocked)
			{
				f_mutexUnlock( m_hMutex);
			}
		}
	
		F_MUTEX					m_hMutex;
		FLMUINT					m_uiOpenThreshold;
		FLMUINT					m_uiMaxAvailTime;
		IF_FileHdl *			m_pFirstUsed;
		IF_FileHdl *			m_pLastUsed;
		FLMUINT					m_uiNumUsed;
		IF_FileHdl *			m_pFirstAvail;
		IF_FileHdl *			m_pLastAvail;
		FLMUINT					m_uiNumAvail;
		FLMBOOL					m_bIsSetup;
		FLMUINT					m_uiFileIdCounter;
	};
	#endif

	/****************************************************************************
	Desc:
	****************************************************************************/
	
	#if defined( FLM_WIN)
	
		typedef struct
		{
			 HANDLE					findHandle;
			 WIN32_FIND_DATA		findBuffer;
			 char 	   			szSearchPath[ F_PATH_MAX_SIZE];
			 FLMUINT					uiSearchAttrib;
		} F_IO_FIND_DATA;
	
	#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	
		typedef struct
		{
			mode_t		mode_flag;
			struct stat	FileStat;
			char 			name[ F_PATH_MAX_SIZE+1];
			char			search_path[ F_PATH_MAX_SIZE+1];
			char			full_path[ F_PATH_MAX_SIZE];
			char			pattern_str[ F_PATH_MAX_SIZE];
			char			dirpath[ F_PATH_MAX_SIZE];
			glob_t      globbuf;
		} F_IO_FIND_DATA;
		
	#elif defined( FLM_RING_ZERO_NLM)
	
		typedef struct
		{
			LONG									lVolumeNumber;
			LONG									lDirectoryNumber;
			LONG									lCurrentEntryNumber;
			struct DirectoryStructure *	pCurrentItem;
			char									ucTempBuffer[ F_FILENAME_SIZE];
		} F_IO_FIND_DATA;
		
	#else
	
		#error Platform not supported
	
	#endif

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_DirHdl : public IF_DirHdl
	{
	public:
	
		F_DirHdl();
	
		virtual ~F_DirHdl();
	
		RCODE FLMAPI next( void);
	
		const char * FLMAPI currentItemName( void);
	
		void FLMAPI currentItemPath(
			char *	pszPath);
	
			FLMUINT64 FLMAPI currentItemSize( void);
	
		FLMBOOL FLMAPI currentItemIsDir( void);
	
	private:
	
		RCODE FLMAPI openDir(
			const char *	pszDirName,
			const char *	pszPattern);
	
		RCODE FLMAPI createDir(
			const char *	pszDirName);
	
		RCODE FLMAPI removeDir(
			const char *	pszDirPath);
	
		char					m_szDirectoryPath[ F_PATH_MAX_SIZE];
		char					m_szPattern[ F_PATH_MAX_SIZE];
		RCODE					m_rc;
		FLMBOOL				m_bFirstTime;
		FLMBOOL				m_bFindOpen;
		FLMUINT				m_uiAttrib;
		F_IO_FIND_DATA		m_FindData;
	#ifndef FLM_RING_ZERO_NLM
		char					m_szFileName[ F_PATH_MAX_SIZE];
	#endif
		
		friend class F_FileSystem;
	};

	/****************************************************************************
	Desc: XML
	****************************************************************************/

	typedef struct xmlChar
	{
		FLMBYTE		ucFlags;
	} XMLCHAR;
	
	class F_XML : public IF_XML
	{
	public:
	
		F_XML();
	
		virtual ~F_XML();
		
		RCODE FLMAPI setup( void);
	
		FLMBOOL FLMAPI isPubidChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isQuoteChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isWhitespace(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isExtender(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isCombiningChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isNameChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isNCNameChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isIdeographic(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isBaseChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isDigit(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isLetter(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isNameValid(
			FLMUNICODE *	puzName,
			FLMBYTE *		pszName);
	
	private:
	
		void setCharFlag(
			FLMUNICODE		uLowChar,
			FLMUNICODE		uHighChar,
			FLMUINT16		ui16Flag);
	
		XMLCHAR *			m_pCharTable;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileSystem : public IF_FileSystem
	{
	public:

		F_FileSystem()
		{
			m_bCanDoAsync = FALSE;
		}

		virtual ~F_FileSystem()
		{
		}
		
		RCODE setup( void);

		FLMINT FLMAPI AddRef( void)
		{
			return( f_atomicInc( &m_refCnt));
		}

		FLMINT FLMAPI Release( void)
		{
			FLMINT		iRefCnt = f_atomicDec( &m_refCnt);
			
			if( !iRefCnt)
			{
				delete this;
			}
			
			return( iRefCnt);
		}
		
		RCODE FLMAPI createFile(
			const char *			pszFileName,
			FLMUINT					uiIoFlags,
			IF_FileHdl **			ppFile);

		RCODE FLMAPI createUniqueFile(
			char *					pszPath,
			const char *			pszFileExtension,
			FLMUINT					uiIoFlags,
			IF_FileHdl **			ppFile);

		RCODE FLMAPI createLockFile(
			const char *			pszPath,
			IF_FileHdl **			ppLockFileHdl);
			
		RCODE FLMAPI openFile(
			const char *			pszFileName,
			FLMUINT					uiIoFlags,
			IF_FileHdl **			ppFile);

		RCODE FLMAPI openDir(
			const char *			pszDirName,
			const char *			pszPattern,
			IF_DirHdl **			ppDir);

		RCODE FLMAPI createDir(
			const char *			pszDirName);

		RCODE FLMAPI removeDir(
			const char *			pszDirName,
			FLMBOOL					bClear = FALSE);

		RCODE FLMAPI doesFileExist(
			const char *			pszFileName);

		FLMBOOL FLMAPI isDir(
			const char *			pszFileName);

		RCODE FLMAPI getFileTimeStamp(
			const char *			pszFileName,
			FLMUINT *				puiTimeStamp);

		RCODE FLMAPI deleteFile(
			const char *			pszFileName);

		RCODE FLMAPI deleteMultiFileStream(
			const char *			pszDirectory,
			const char *			pszBaseName);
			
		RCODE FLMAPI copyFile(
			const char *			pszSrcFileName,
			const char *			pszDestFileName,
			FLMBOOL					bOverwrite,
			FLMUINT64 *				pui64BytesCopied);

		RCODE FLMAPI copyPartialFile(
			IF_FileHdl *			pSrcFileHdl,
			FLMUINT64				ui64SrcOffset,
			FLMUINT64				ui64SrcSize,
			IF_FileHdl *			pDestFileHdl,
			FLMUINT64				ui64DestOffset,
			FLMUINT64 *				pui64BytesCopiedRV);
		
		RCODE FLMAPI renameFile(
			const char *			pszFileName,
			const char *			pszNewFileName);

		void FLMAPI pathParse(
			const char *			pszPath,
			char *					pszServer,
			char *					pszVolume,
			char *					pszDirPath,
			char *					pszFileName);

		RCODE FLMAPI pathReduce(
			const char *			pszSourcePath,
			char *					pszDestPath,
			char *					pszString);

		RCODE FLMAPI pathAppend(
			char *					pszPath,
			const char *			pszPathComponent);

		RCODE FLMAPI pathToStorageString(
			const char *			pszPath,
			char *					pszString);

		void FLMAPI pathCreateUniqueName(
			FLMUINT *				puiTime,
			char *					pszFileName,
			const char *			pszFileExt,
			FLMBYTE *				pHighChars,
			FLMBOOL					bModext);

		FLMBOOL FLMAPI doesFileMatch(
			const char *			pszFileName,
			const char *			pszTemplate);

		RCODE FLMAPI getSectorSize(
			const char *			pszFileName,
			FLMUINT *				puiSectorSize);

		RCODE FLMAPI setReadOnly(
			const char *			pszFileName,
			FLMBOOL					bReadOnly);

		FLMBOOL FLMAPI canDoAsync( void);
			
	private:

		RCODE removeEmptyDir(
			const char *			pszDirName);

	#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
		RCODE renameSafe(
			const char *			pszSrcFile,
			const char *			pszDestFile);

		RCODE targetIsDir(
			const char	*			pszPath,
			FLMBOOL *				pbIsDir);
	#endif

		FLMBOOL				m_bCanDoAsync;
	};
	
	/****************************************************************************
	Desc: Logging
	****************************************************************************/

	void flmDbgLogInit( void);
	void flmDbgLogExit( void);
	void flmDbgLogFlush( void);

	/****************************************************************************
	Desc: Logger client
	****************************************************************************/
	RCODE f_loggerInit( void);

	void f_loggerShutdown( void);

	/****************************************************************************
	Desc:	Misc.
	****************************************************************************/
	FLMUINT f_getFSBlockSize(
		FLMBYTE *			pszFileName);
		
	#if defined( FLM_LINUX)

		void f_setupLinuxKernelVersion( void);

		void f_getLinuxKernelVersion(
			FLMUINT *		puiMajor,
			FLMUINT *		puiMinor,
			FLMUINT *		puiRevision);
			
		FLMUINT f_getLinuxMaxFileSize( void);
		
		void f_getLinuxMemInfo(
			FLMUINT64 *		pui64TotalMem,
			FLMUINT64 *		pui64AvailMem);
	#endif

	void f_memoryInit( void);
	
	void f_memoryCleanup( void);
	
	RCODE f_netwareStartup( void);
	
	void f_netwareShutdown( void);
	
	void f_initFastCheckSum( void);
	
	RCODE f_initCRCTable( void);

	void f_freeCRCTable( void);
	
	RCODE f_allocFileSystem(
		IF_FileSystem **	ppFileSystem);
		
	RCODE f_allocThreadMgr(
		IF_ThreadMgr **	ppThreadMgr);
	
	RCODE f_allocFileHdl(
		F_FileHdl **		ppFileHdl);
	
	RCODE f_allocDirHdl(
		F_DirHdl **			ppDirHdl);
		
	IF_ThreadMgr * f_getThreadMgrPtr( void);
	
	RCODE f_verifyMetaphoneRoutines( void);
	
	RCODE f_verifyDiskStructOffsets( void);

	RCODE f_initCharMappingTables( void);
	
	void f_freeCharMappingTables( void);
	
	IF_XML * f_getXmlObjPtr( void);
		
	RCODE f_netwareRemoveDir( 
		const char *		pszDirName);
	
	RCODE f_netwareTestIfFileExists(
		const char *		pPath);
		
	RCODE f_netwareDeleteFile(
		const char *		pPath);
		
	RCODE f_netwareRenameFile(
		const char *		pOldFilePath,
		const char *		pNewFilePath);
	
#endif	// FTKSYS_H
