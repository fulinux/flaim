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

	/****************************************************************************
	Desc:		NLM
	****************************************************************************/
	#if defined( FLM_NLM)
	
		#if defined( FLM_WATCOM_NLM)
			#pragma warning 007 9
	
			// Disable "Warning! W549: col(XX) 'sizeof' operand contains
			// compiler generated information"
			
			#pragma warning 549 9
			
			// Disable "Warning! W656: col(XX) define this function inside its class
			// definition (may improve code quality)"
			
			#pragma warning 656 9
			
			// Disable Warning! W555: col(XX) expression for 'while' is always
			// "false"
			
			#pragma warning 555 9
		#endif
		
		#define _POSIX_SOURCE

		#include <stdio.h>
		#include <string.h>
		#include <pthread.h>
		#include <unistd.h>
		#include <errno.h>
		#include <library.h>
		#include <fcntl.h>
		#include <sys/stat.h>
		#include <sys/unistd.h>
		#include <glob.h>
		#include <netware.h>
		#include <semaphore.h>
		#include <malloc.h>
		#include <novsock2.h>

		// The typedef for va_list in stdarg.h do not function properly when
		// a va_list is passed down multiple layers as a pointer (va_list *).
		// Therefore, the following definitions/typedefs were taken from a
		// "fixed" version of stdarg.h implemented by DS.

		// typedef unsigned long f_va_list;
		
		#define f_argsize(x) \
			((sizeof(x)+sizeof(int)-1) & ~(sizeof(int)-1))
			
		#define f_va_start(ap, parmN) \
			((void)((ap) = (unsigned long)&(parmN) + f_argsize(parmN)))
			
		#define f_va_arg(ap, type) \
			(*(type *)(((ap) += f_argsize(type)) - (f_argsize(type))))
			
		#define f_va_end(ap) ((void)0)

		#ifndef _SIZE_T
			#define _SIZE_T
			typedef unsigned int size_t;
		#endif

		#ifndef _WCHAR_T
			#define _WCHAR_T
			typedef unsigned short wchar_t;
		#endif

		#ifndef WCHAR
			#define WCHAR wchar_t
		#endif

		#ifndef LONG
			#define LONG unsigned long	
		#endif

		#ifndef BYTE
			#define BYTE unsigned char
		#endif

		#ifndef UINT
			#define UINT	unsigned int
		#endif
		
		#define F_NETWARE_SECTOR_SIZE			512
		
		FINLINE void * f_getNLMHandle( void)
		{
			return( getnlmhandle());
		}
	
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
		#pragma pack( pop, enter_windows)
		
		// Conversion from XXX to YYY, possible loss of data
		#pragma warning( disable : 4244) 
	
		// Local variable XXX may be used without having been initialized
		#pragma warning( disable : 4701)
	
		// Function XXX not inlined
		#pragma warning( disable : 4710) 
		
		#define ENDLINE			ENDLINE_CRLF
		#define f_va_start		va_start
		#define f_va_arg			va_arg
		#define f_va_end			va_end
		
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
			#include <stdio.h>
			#include <fcntl.h>
			#include <sys/vminfo.h>
		#endif

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
			#include <libkern/OSAtomic.h>
		#endif

		#define f_va_start			va_start
		#define f_va_arg				va_arg
		#define f_va_end				va_end

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
	#if defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
	extern "C" FLMINT32 sparc_atomic_add_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
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
	extern "C" FLMINT32 posix_atomic_add_32(
		volatile FLMINT32 *		piTarget,
		FLMINT32						iDelta);
	#endif
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	#if defined( FLM_UNIX)
	extern "C" FLMINT32 posix_atomic_xchg_32(
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
										Process ID Functions
	****************************************************************************/

	#if defined( FLM_WIN)

		FINLINE FLMUINT f_getpid( void)
		{ 
			return _getpid();
		}

	#elif defined( FLM_UNIX)

		pid_t getpid( void);

		FINLINE FLMUINT f_getpid( void)
		{ 
			return getpid();
		}

	#elif defined( FLM_NLM)
	
		FINLINE FLMUINT f_getpid() 
		{ 
			return( (FLMUINT)f_getNLMHandle());
		}
		
	#else
		#error "Unsupported Platform"
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

	typedef struct
	{
		FLMBYTE *	pszDestStr;
	} F_SPRINTF_INFO;

	void f_sprintfProcessFieldInfo(
		FLMBYTE **			ppszFormat,
		FLMUINT *			puiWidth,
		FLMUINT *			puiPrecision,
		FLMUINT *			puiFlags,
		f_va_list *			args);

	void f_sprintfStringFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfCharFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfErrorFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfNotHandledFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfNumberFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);
		
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
	
		friend class F_IOBufferMgr;
	};

	/****************************************************************************
	Desc:
	*****************************************************************************/
	class F_DynaBuf : public IF_DynaBuf
	{
	public:
	
		F_DynaBuf(
			FLMBYTE *		pucBuffer,
			FLMUINT			uiBufferSize)
		{
			m_pucBuffer = pucBuffer;
			m_uiBufferSize = uiBufferSize;
			m_uiOffset = 0;
			m_bAllocatedBuffer = FALSE;
		}
		
		virtual ~F_DynaBuf()
		{
			if( m_bAllocatedBuffer)
			{
				f_free( &m_pucBuffer);
			}
		}
		
		FINLINE void FLMAPI truncateData(
			FLMUINT			uiSize)
		{
			if( uiSize < m_uiOffset)
			{
				m_uiOffset = uiSize;
			}
		}
		
		FINLINE RCODE FLMAPI allocSpace(
			FLMUINT		uiSize,
			void **		ppvPtr)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( m_uiOffset + uiSize >= m_uiBufferSize)
			{
				if( RC_BAD( rc = resizeBuffer( m_uiOffset + uiSize + 512)))
				{
					goto Exit;
				}
			}
			
			*ppvPtr = &m_pucBuffer[ m_uiOffset];
			m_uiOffset += uiSize;
			
		Exit:
		
			return( rc);
		}
		
		FINLINE RCODE FLMAPI appendData(
			const void *		pvData,
			FLMUINT				uiSize)
		{
			RCODE		rc = NE_FLM_OK;
			void *	pvTmp;
			
			if( RC_BAD( rc = allocSpace( uiSize, &pvTmp)))
			{
				goto Exit;
			}
	
			if( uiSize == 1)
			{
				*((FLMBYTE *)pvTmp) = *((FLMBYTE *)pvData);
			}
			else
			{
				f_memcpy( pvTmp, pvData, uiSize);
			}
			
		Exit:
		
			return( rc);
		}
			
		FINLINE RCODE FLMAPI appendByte(
			FLMBYTE		ucChar)
		{
			RCODE			rc = NE_FLM_OK;
			FLMBYTE *	pucTmp;
			
			if( RC_BAD( rc = allocSpace( 1, (void **)&pucTmp)))
			{
				goto Exit;
			}
			
			*pucTmp = ucChar;
			
		Exit:
		
			return( rc);
		}
		
		FINLINE RCODE FLMAPI appendUniChar(
			FLMUNICODE	uChar)
		{
			RCODE				rc = NE_FLM_OK;
			FLMUNICODE *	puTmp;
			
			if( RC_BAD( rc = allocSpace( sizeof( FLMUNICODE), (void **)&puTmp)))
			{
				goto Exit;
			}
			
			*puTmp = uChar;
			
		Exit:
		
			return( rc);
		}
		
		FINLINE FLMBYTE * FLMAPI getBufferPtr( void)
		{
			return( m_pucBuffer);
		}
		
		FINLINE FLMUNICODE * FLMAPI getUnicodePtr( void)
		{
			if( m_uiOffset >= sizeof( FLMUNICODE))
			{
				return( (FLMUNICODE *)m_pucBuffer);
			}
			
			return( NULL);
		}
		
		FINLINE FLMUINT FLMAPI getUnicodeLength( void)
		{
			if( m_uiOffset <= sizeof( FLMUNICODE))
			{
				return( 0);
			}
			
			return( (m_uiOffset >> 1) - 1);
		}
		
		FINLINE FLMUINT FLMAPI getDataLength( void)
		{
			return( m_uiOffset);
		}
		
		FINLINE RCODE FLMAPI copyFromBuffer(
			IF_DynaBuf *		pSource)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( RC_BAD( rc = resizeBuffer( 
				((F_DynaBuf *)pSource)->m_uiBufferSize)))
			{
				goto Exit;
			}
			
			if( (m_uiOffset = ((F_DynaBuf *)pSource)->m_uiOffset) != 0)
			{
				f_memcpy( m_pucBuffer, ((F_DynaBuf *)pSource)->m_pucBuffer, 
					((F_DynaBuf *)pSource)->m_uiOffset);
			}
			
		Exit:
			
			return( rc);
		}		
		
	private:
	
		FINLINE RCODE resizeBuffer(
			FLMUINT		uiNewSize)
		{
			RCODE	rc = NE_FLM_OK;
			
			if( !m_bAllocatedBuffer)
			{
				if( uiNewSize > m_uiBufferSize)
				{
					FLMBYTE *		pucOriginalBuf = m_pucBuffer;
					
					if( RC_BAD( rc = f_alloc( uiNewSize, &m_pucBuffer)))
					{
						m_pucBuffer = pucOriginalBuf;
						goto Exit;
					}
					
					m_bAllocatedBuffer = TRUE;
					
					if( m_uiOffset)
					{
						f_memcpy( m_pucBuffer, pucOriginalBuf, m_uiOffset);
					}
				}
			}
			else
			{
				if( RC_BAD( rc = f_realloc( uiNewSize, &m_pucBuffer)))
				{
					goto Exit;
				}
				
				if( uiNewSize < m_uiOffset)
				{
					m_uiOffset = uiNewSize;
				}
			}
			
			m_uiBufferSize = uiNewSize;
			
		Exit:
		
			return( rc);
		}
	
		FLMBOOL		m_bAllocatedBuffer;
		FLMBYTE *	m_pucBuffer;
		FLMUINT		m_uiBufferSize;
		FLMUINT		m_uiOffset;
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
						pvBuffer, TRUE, puiBytesReadRV));
			}
			else
			{
				return( read( ui64ReadOffset, uiBytesToRead, 
						pvBuffer, puiBytesReadRV));
			}
		}
	
		FINLINE RCODE FLMAPI sectorWrite(
			FLMUINT64		ui64WriteOffset,
			FLMUINT			uiBytesToWrite,
			const void *	pvBuffer,
			FLMUINT			uiBufferSize,
			void *			pvBufferObj,
			FLMUINT *		puiBytesWrittenRV,
			FLMBOOL			bZeroFill = TRUE)
		{
			F_UNREFERENCED_PARM( uiBufferSize);
			
			if (m_bDoDirectIO)
			{
				return( directWrite( ui64WriteOffset, uiBytesToWrite,
						pvBuffer, (F_IOBuffer *)pvBufferObj, TRUE,
						bZeroFill, puiBytesWrittenRV));
			}
			else
			{
				f_assert( pvBufferObj == NULL);
				return( write( ui64WriteOffset, uiBytesToWrite, pvBuffer,
									puiBytesWrittenRV));
			}
		}
	
		FINLINE FLMBOOL FLMAPI canDoAsync( void)
		{
			return( m_bCanDoAsync);
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
		
		FINLINE void FLMAPI setBlockSize(
			FLMUINT	uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
		}
		
		FINLINE FLMUINT FLMAPI getBlockSize( void)
		{
			return( m_uiBlockSize);
		}
		
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
			FLMBOOL			bBuffHasFullSectors,
			FLMUINT *		puiBytesRead);
	
		RCODE directWrite(
			FLMUINT64		uiOffset,
			FLMUINT			uiLength,
			const void *	pvBuffer,
			F_IOBuffer *	pBufferObj,
			FLMBOOL			bBuffHasFullSectors,
			FLMBOOL			bZeroFill,
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
		FLMUINT				m_uiBlockSize;
		FLMUINT				m_uiBytesPerSector;
		FLMUINT64			m_ui64NotOnSectorBoundMask;
		FLMUINT64			m_ui64GetSectorBoundMask;
		FLMBOOL				m_bDoDirectIO;
		FLMUINT				m_uiExtendSize;
		FLMUINT				m_uiMaxAutoExtendSize;
		FLMBYTE *			m_pucAlignedBuff;
		FLMUINT				m_uiAlignedBuffSize;
		FLMUINT64			m_ui64CurrentPos;
		FLMBOOL				m_bCanDoAsync;
		OVERLAPPED			m_Overlapped;
		
		friend class F_FileSystem;
		friend class F_MultiFileHdl;
	};
	#endif

	/***************************************************************************
	Desc:
	***************************************************************************/
	#if defined( FLM_UNIX) || defined( FLM_NLM) 
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
			FLMUINT				uiBufferSize,
			void *				pvBufferObj,
			FLMUINT *			puiBytesWrittenRV,
			FLMBOOL				bZeroFill = TRUE)
		{
			if( m_bDoDirectIO)
			{
				return( directWrite( ui64WriteOffset, uiBytesToWrite, 
					pvBuffer, uiBufferSize, (F_IOBuffer *)pvBufferObj, 
					puiBytesWrittenRV, TRUE, bZeroFill));
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
		
		FINLINE void FLMAPI setBlockSize(
			FLMUINT				uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
		}
		
		FINLINE FLMUINT FLMAPI getBlockSize( void)
		{
			return( m_uiBlockSize);
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
	
		FINLINE FLMUINT64 roundUpToSectorMultiple(
			FLMUINT64			ui64Bytes)
		{
			return( (ui64Bytes + m_ui64NotOnSectorBoundMask) &
						m_ui64GetSectorBoundMask);
		}
	
		FINLINE FLMUINT64 getSectorStartOffset(
			FLMUINT64			ui64Offset)
		{
			return( ui64Offset & m_ui64GetSectorBoundMask);
		}
	
		RCODE directRead(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,	
			void *				pvBuffer,
			FLMBOOL				bBuffHasFullSectors,
			FLMUINT *			puiBytesRead);
	
		RCODE directWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			FLMUINT				uiBufferSize,
			F_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV,
			FLMBOOL				bBuffHasFullSectors,
			FLMBOOL				bZeroFill);
		
		RCODE allocAlignedBuffer( void);
		
		FLMBOOL					m_bFileOpened;
		FLMBOOL					m_bDeleteOnRelease;
		FLMBOOL					m_bOpenedReadOnly;
		FLMBOOL					m_bOpenedExclusive;
		char *					m_pszFileName;
		int				   	m_fd;
		FLMUINT					m_uiBlockSize;
		FLMUINT					m_uiBytesPerSector;
		FLMUINT64				m_ui64NotOnSectorBoundMask;
		FLMUINT64				m_ui64GetSectorBoundMask;
		FLMUINT64				m_ui64CurrentPos;
		FLMUINT					m_uiExtendSize;
		FLMBOOL					m_bCanDoAsync;
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
	
	#elif defined( FLM_UNIX) || defined( FLM_NLM)
	
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
		FLMUINT32			m_ui32RefCount;
		RCODE					m_rc;
		FLMBOOL				m_bFirstTime;
		FLMBOOL				m_bFindOpen;
		FLMBOOL				m_EOF;
		char					m_szFileName[ F_PATH_MAX_SIZE];
		FLMUINT				m_uiAttrib;
		F_IO_FIND_DATA		m_FindData;
		
		friend class F_FileSystem;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IStream : public IF_IStream
	{
	public:
	
		F_IStream();
	
		virtual ~F_IStream();
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_OStream : public IF_OStream
	{
	public:
	
		F_OStream();
	
		virtual ~F_OStream();
	
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_PosIStream : public IF_PosIStream
	{
	public:
	
		F_PosIStream();
	
		virtual ~F_PosIStream();
	
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferIStream : public IF_BufferIStream
	{
	public:
	
		F_BufferIStream()
		{
			m_pucBuffer = NULL;
			m_uiBufferLen = 0;
			m_uiOffset = 0;
			m_bAllocatedBuffer = FALSE;
			m_bIsOpen = FALSE;
		}
	
		virtual ~F_BufferIStream();
	
		RCODE FLMAPI open(
			const char *		pucBuffer,
			FLMUINT				uiLength,
			char **				ppucAllocatedBuffer = NULL);
	
		FINLINE FLMUINT64 FLMAPI totalSize( void)
		{
			f_assert( m_bIsOpen);
			return( m_uiBufferLen);
		}
	
		FINLINE FLMUINT64 FLMAPI remainingSize( void)
		{
			f_assert( m_bIsOpen);
			return( m_uiBufferLen - m_uiOffset);
		}
	
		RCODE FLMAPI close( void);
	
		FINLINE RCODE FLMAPI positionTo(
			FLMUINT64		ui64Position)
		{
			f_assert( m_bIsOpen);
	
			if( ui64Position < m_uiBufferLen)
			{
				m_uiOffset = (FLMUINT)ui64Position;
			}
			else
			{
				m_uiOffset = m_uiBufferLen;
			}
	
			return( NE_FLM_OK);
		}
	
		FINLINE FLMUINT64 FLMAPI getCurrPosition( void)
		{
			f_assert( m_bIsOpen);
			return( m_uiOffset);
		}
	
		FINLINE void FLMAPI truncate(
			FLMUINT64		ui64Offset)
		{
			f_assert( m_bIsOpen);
			f_assert( ui64Offset >= m_uiOffset);
			f_assert( ui64Offset <= m_uiBufferLen);
			
			m_uiBufferLen = (FLMUINT)ui64Offset;
		}
	
		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
			
		FINLINE const FLMBYTE * getBuffer( void)
		{
			f_assert( m_bIsOpen);
			return( m_pucBuffer);
		}
		
		FINLINE const FLMBYTE * FLMAPI getBufferAtCurrentOffset( void)
		{
			f_assert( m_bIsOpen);
			return( m_pucBuffer ? &m_pucBuffer[ m_uiOffset] : NULL);
		}
		
		FINLINE FLMBOOL isOpen( void)
		{
			return( m_bIsOpen);
		}
	
	private:
	
		const FLMBYTE *	m_pucBuffer;
		FLMUINT				m_uiBufferLen;
		FLMUINT				m_uiOffset;
		FLMBOOL				m_bAllocatedBuffer;
		FLMBOOL				m_bIsOpen;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileIStream : public F_PosIStream
	{
	public:
	
		F_FileIStream()
		{
			m_pFileHdl = NULL;
			m_ui64FileOffset = 0;
		}
	
		virtual ~F_FileIStream()
		{
			if( m_pFileHdl)
			{
				m_pFileHdl->Release();
			}
		}
	
		RCODE FLMAPI open(
			const char *	pszPath);
	
		RCODE FLMAPI close( void);
	
		RCODE FLMAPI positionTo(
			FLMUINT64		ui64Position);
	
		FLMUINT64 FLMAPI totalSize( void);
	
		FLMUINT64 FLMAPI remainingSize( void);
	
		FLMUINT64 FLMAPI getCurrPosition( void);
	
		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
	
	private:
	
		IF_FileHdl *		m_pFileHdl;
		FLMUINT64			m_ui64FileOffset;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferedIStream : public F_PosIStream
	{
	public:
	
		F_BufferedIStream()
		{
			m_pIStream = NULL;
			m_pucBuffer = NULL;
		}
	
		virtual ~F_BufferedIStream()
		{
			close();
		}
	
		RCODE FLMAPI open(
			IF_IStream *		pIStream,
			FLMUINT				uiBufferSize);
	
		RCODE FLMAPI read(
			void *				pvBuffer,
			FLMUINT				uiBytesToRead,
			FLMUINT *			puiBytesRead);
	
		RCODE FLMAPI close( void);
	
		FINLINE FLMUINT64 FLMAPI totalSize( void)
		{
			if (!m_pIStream)
			{
				f_assert( 0);
				return( 0);
			}
	
			return( m_uiBytesAvail);
		}
	
		FINLINE FLMUINT64 FLMAPI remainingSize( void)
		{
			if( !m_pIStream)
			{
				f_assert( 0);
				return( 0);
			}
	
			return( m_uiBytesAvail - m_uiBufferOffset);
		}
	
		FINLINE RCODE FLMAPI positionTo(
			FLMUINT64		ui64Position)
		{
			if( !m_pIStream)
			{
				f_assert( 0);
				return( RC_SET( NE_FLM_ILLEGAL_OP));
			}
	
			if( ui64Position < m_uiBytesAvail)
			{
				m_uiBufferOffset = (FLMUINT)ui64Position;
			}
			else
			{
				m_uiBufferOffset = m_uiBytesAvail;
			}
	
			return( NE_FLM_OK);
		}
	
		FINLINE FLMUINT64 FLMAPI getCurrPosition( void)
		{
			if( !m_pIStream)
			{
				f_assert( 0);
				return( 0);
			}
	
			return( m_uiBufferOffset);
		}
	
	private:
	
		IF_IStream *			m_pIStream;
		FLMBYTE *				m_pucBuffer;
		FLMUINT					m_uiBufferSize;
		FLMUINT					m_uiBufferOffset;
		FLMUINT					m_uiBytesAvail;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferedOStream : public F_OStream
	{
	public:
	
		F_BufferedOStream()
		{
			m_pOStream = NULL;
			m_pucBuffer = NULL;
		}
	
		virtual ~F_BufferedOStream()
		{
			close();
		}
	
		RCODE FLMAPI open(
			IF_OStream *	pOStream,
			FLMUINT			uiBufferSize);
	
		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);
	
		RCODE FLMAPI close( void);
	
		RCODE FLMAPI flush( void);
	
	private:
	
		IF_OStream *		m_pOStream;
		FLMBYTE *			m_pucBuffer;
		FLMUINT				m_uiBufferSize;
		FLMUINT				m_uiBufferOffset;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileOStream : public F_OStream
	{
	public:
	
		F_FileOStream()
		{
			m_pFileHdl = NULL;
		}
	
		virtual ~F_FileOStream()
		{
			close();
		}
	
		RCODE FLMAPI open(
			const char *	pszFilePath,
			FLMBOOL			bTruncateIfExists);
	
		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);
	
		RCODE FLMAPI close( void);
	
	private:
	
		IF_FileHdl *		m_pFileHdl;
		FLMUINT64			m_ui64FileOffset;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_MultiFileIStream : public F_IStream
	{
	public:
	
		F_MultiFileIStream()
		{
			m_pIStream = NULL;
			m_bOpen = FALSE;
		}
	
		virtual ~F_MultiFileIStream()
		{
			close();
		}
	
		RCODE FLMAPI open(
			const char *	pszDirectory,
			const char *	pszBaseName);
	
		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
	
		RCODE FLMAPI close( void);
	
	private:
	
		RCODE rollToNextFile( void);
	
		IF_IStream *		m_pIStream;
		FLMBOOL				m_bOpen;
		FLMBOOL				m_bEndOfStream;
		FLMUINT				m_uiFileNum;
		FLMUINT64			m_ui64FileOffset;
		char 					m_szDirectory[ F_PATH_MAX_SIZE + 1];
		char	 				m_szBaseName[ F_PATH_MAX_SIZE + 1];
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_MultiFileOStream : public F_OStream
	{
	public:
	
		F_MultiFileOStream()
		{
			m_pOStream = NULL;
			m_bOpen = FALSE;
		}
	
		virtual ~F_MultiFileOStream()
		{
			close();
		}
	
		RCODE create(
			const char *	pszDirectory,
			const char *	pszBaseName,
			FLMUINT			uiMaxFileSize,
			FLMBOOL			bOkToOverwrite);
	
		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);
	
		RCODE FLMAPI close( void);
	
		RCODE processDirectory(
			const char *	pszDirectory,
			const char *	pszBaseName,
			FLMBOOL			bOkToDelete);
	
	private:
	
		RCODE rollToNextFile( void);
	
		F_OStream *		m_pOStream;
		FLMBOOL			m_bOpen;
		FLMUINT			m_uiFileNum;
		FLMUINT64		m_ui64MaxFileSize;
		FLMUINT64		m_ui64FileOffset;
		char 				m_szDirectory[ F_PATH_MAX_SIZE + 1];
		char 				m_szBaseName[ F_PATH_MAX_SIZE + 1];
		
		friend class F_FileSystem;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_CollIStream : public IF_CollIStream
	{
	public:
	
		F_CollIStream()
		{
			m_pIStream = NULL;
			m_uiLanguage = 0;
			m_bMayHaveWildCards = FALSE;
			m_bUnicodeStream = FALSE;
			m_uNextChar = 0;
		}
	
		virtual ~F_CollIStream()
		{
			if( m_pIStream)
			{
				m_pIStream->Release();
			}
		}
	
		RCODE FLMAPI open(
			IF_PosIStream *	pIStream,
			FLMBOOL				bUnicodeStream,
			FLMUINT				uiLanguage,
			FLMUINT				uiCompareRules,
			FLMBOOL				bMayHaveWildCards)
		{
			if( m_pIStream)
			{
				m_pIStream->Release();
			}
	
			m_pIStream = pIStream;
			m_pIStream->AddRef();
			m_uiLanguage = uiLanguage;
			m_uiCompareRules = uiCompareRules;
			m_bCaseSensitive = (uiCompareRules & FLM_COMP_CASE_INSENSITIVE)
									  ? FALSE
									  : TRUE;
			m_bMayHaveWildCards = bMayHaveWildCards;
			m_bUnicodeStream = bUnicodeStream;		
			m_ui64EndOfLeadingSpacesPos = 0;
			return( NE_FLM_OK);
		}
	
		RCODE FLMAPI close( void)
		{
			if( m_pIStream)
			{
				m_pIStream->Release();
				m_pIStream = NULL;
			}
			
			return( NE_FLM_OK);
		}
	
		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead)
		{
			RCODE		rc = NE_FLM_OK;
	
			if( RC_BAD( rc = m_pIStream->read( pvBuffer, 
				uiBytesToRead, puiBytesRead)))
			{
				goto Exit;
			}
	
		Exit:
	
			return( rc);
		}
	
		RCODE FLMAPI read(
			FLMBOOL			bAllowTwoIntoOne,
			FLMUNICODE *	puChar,
			FLMBOOL *		pbCharIsWild,
			FLMUINT16 *		pui16Col,
			FLMUINT16 *		pui16SubCol,
			FLMBYTE *		pucCase);
			
		FINLINE FLMUINT64 FLMAPI totalSize( void)
		{
			if( m_pIStream)
			{
				return( m_pIStream->totalSize());
			}
	
			return( 0);
		}
	
		FINLINE FLMUINT64 FLMAPI remainingSize( void)
		{
			if( m_pIStream)
			{
				return( m_pIStream->remainingSize());
			}
	
			return( 0);
		}
	
		FINLINE RCODE FLMAPI positionTo(
			FLMUINT64)
		{
			return( RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED));
		}
	
		FINLINE RCODE FLMAPI positionTo(
			F_CollStreamPos *	pPos)
		{
			
			// Should never be able to position back to before the
			// leading spaces.
			
			m_uNextChar = pPos->uNextChar;
			flmAssert( pPos->ui64Position >= m_ui64EndOfLeadingSpacesPos);
			return( m_pIStream->positionTo( pPos->ui64Position));
		}
	
		FINLINE FLMUINT64 FLMAPI getCurrPosition( void)
		{
			flmAssert( 0);
			return( 0);
		}
	
		void FLMAPI getCurrPosition(
			F_CollStreamPos *		pPos);
	
	private:
	
		FINLINE RCODE readCharFromStream(
			FLMUNICODE *		puChar)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( m_bUnicodeStream)
			{
				if( RC_BAD( rc = m_pIStream->read( puChar, sizeof( FLMUNICODE), NULL)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = f_readUTF8CharAsUnicode( 
					m_pIStream, puChar)))
				{
					goto Exit;
				}
			}
			
		Exit:
		
			return( rc);
		}
			
		IF_PosIStream *	m_pIStream;
		FLMUINT				m_uiLanguage;
		FLMBOOL				m_bCaseSensitive;
		FLMUINT				m_uiCompareRules;
		FLMUINT64			m_ui64EndOfLeadingSpacesPos;
		FLMBOOL				m_bMayHaveWildCards;
		FLMBOOL				m_bUnicodeStream;
		FLMUNICODE			m_uNextChar;
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
	Desc: Logging
	****************************************************************************/

	void flmDbgLogInit( void);
	void flmDbgLogExit( void);
	void flmDbgLogFlush( void);

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
	
	RCODE f_checkErrorCodeTables( void);

	RCODE f_allocFileSystem(
		IF_FileSystem **	ppFileSystem);
		
	RCODE f_allocThreadMgr(
		IF_ThreadMgr **	ppThreadMgr);
	
	RCODE f_allocFileHdl(
		F_FileHdl **		ppFileHdl);
	
	RCODE f_allocDirHdl(
		F_DirHdl **			ppDirHdl);
		
	IF_FileSystem * f_getFileSysPtr( void);

	IF_ThreadMgr * f_getThreadMgrPtr( void);
	
	RCODE f_verifyMetaphoneRoutines( void);
	
	RCODE f_initCharMappingTables( void);
	
	void f_freeCharMappingTables( void);
	
	IF_XML * f_getXmlObjPtr( void);
	
#endif	// FTKSYS_H
