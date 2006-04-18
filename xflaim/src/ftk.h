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

#ifndef FTK_H
#define FTK_H

	#ifndef FLM_PLATFORM_CONFIGURED
		#error Platform not configured
	#endif

	/****************************************************************************
	Desc:		Internal return code macros
	****************************************************************************/
	#ifdef FLM_DEBUG
		RCODE	flmMakeErr(
			RCODE				rc,
			const char *	pszFile,
			int				iLine,
			FLMBOOL			bAssert);
			
		#define RC_SET( rc) \
			flmMakeErr( rc, __FILE__, __LINE__, FALSE)
			
		#define RC_SET_AND_ASSERT( rc) \
			flmMakeErr( rc, __FILE__, __LINE__, TRUE)
			
		#define RC_UNEXPECTED_ASSERT( rc) \
			flmMakeErr( rc, __FILE__, __LINE__, TRUE)
	#else
		#define RC_SET( rc)							(rc)
		#define RC_SET_AND_ASSERT( rc)			(rc)
		#define RC_UNEXPECTED_ASSERT( rc)
	#endif

	#define F_SEM_WAITFOREVER			(0xFFFFFFFF)

	/****************************************************************************
	Desc:		NLM
	****************************************************************************/
	#if defined( FLM_NLM)
	
		#if defined( FLM_WATCOM_NLM)
			#pragma warning 007 9
	
			// Disable "Warning! W549: col(XX) 'sizeof' operand contains
			// compiler generated information"
			
			#pragma warning 549 9
			
			// Disable "Warning! W656: col(1) define this function inside its class
			// definition (may improve code quality)"
			
			#pragma warning 656 9
		#endif

		#include <stdio.h>
		#include <stdlib.h>
		#include <string.h>
		#include <pthread.h>
		#include <unistd.h>
		#include <errno.h>
		#include <library.h>
		#include <fcntl.h>
		#include <sys/stat.h>
		#include <sys/unistd.h>
		#include <glob.h>

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
			
		#define f_va_end(ap) ((void)0)
		#define FSTATIC

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
	
		typedef void * 			MUTEX;
		typedef MUTEX				F_MUTEX;
		typedef MUTEX *			F_MUTEX_p;

		#define F_MUTEX_NULL		0
		
		typedef void * 			SEMAPHORE;
		typedef unsigned long 	ERROR;

		extern "C" SEMAPHORE kSemaphoreAlloc(
			BYTE *		pSemaName,
			UINT			SemaCount);
			
		extern "C" ERROR kSemaphoreFree(
			SEMAPHORE	SemaHandle);
			
		extern "C" ERROR kSemaphoreWait(
			SEMAPHORE	SemaHandle);
			
		extern "C" ERROR kSemaphoreTimedWait(
			SEMAPHORE	SemaHandle,
			UINT			MilliSecondTimeOut);
			
		extern "C" ERROR kSemaphoreSignal(
			SEMAPHORE	SemaHandle);
			
		extern "C" UINT kSemaphoreExamineCount(
			SEMAPHORE	SemaHandle);
	
		extern "C" MUTEX kMutexAlloc(
			BYTE *		MutexName);
			
		extern "C" ERROR kMutexFree(
			MUTEX			MutexHandle);
			
		extern "C" ERROR kMutexLock(
			MUTEX			MutexHandle);
			
		extern "C" ERROR kMutexUnlock(
			MUTEX			MutexHandle);
	
		// External Netware Symbols
		
		extern "C" FLMUINT f_getNLMHandle( void);
		
		extern "C" RCODE f_netwareStartup( void);

		extern "C" void f_netwareShutdown( void);
			
		#define f_stricmp(str1,str2) \
			strcasecmp((char *)(str1),(char *)(str2))

		#define f_strnicmp(str1,str2,size_t) \
			strncasecmp((char *)(str1),(char *)(str2),size_t)

		#define f_memmove( dest, src, len) \
			memmove( (void*)(dest), (void*)(src), len)

		#define f_memset( src, chr, size) \
			memset((void  *)(src),(chr),(size_t)(size))

		#define f_memcmp( str1, str2, length) \
			memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))

		#define f_strcat( dest, src) \
			strcat( (char*)(dest), (char*)(src))

		#define f_strchr( str, value) \
			strchr( (char*)str, (int)value)

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
			(char *)strstr( (char*)(str1), (char*)(str2))

		#define f_strncat( str1, str2, n) \
			strncat( (char *)(str1), (char *)(str2), n)

		#define f_strupr( str) \
			strupr( (char *)(str))

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
		#pragma pack( pop, enter_windows)
		
		// Conversion from XXX to YYY, possible loss of data
		#pragma warning( disable : 4244) 
	
		// Local variable XXX may be used without having been initialized
		#pragma warning( disable : 4701)
	
		// Function XXX not inlined
		#pragma warning( disable : 4710) 
		
		#define FSTATIC			static

		#define ENDLINE			ENDLINE_CRLF
		#define f_va_list			va_list
		#define f_va_start		va_start
		#define f_va_arg			va_arg
		#define f_va_end			va_end

		typedef struct
		{
			FLMATOMIC						locked;
	#ifdef FLM_DEBUG
			FLMUINT							uiThreadId;
			FLMATOMIC						lockedCount;
			FLMATOMIC						waitCount;
	#endif
		} F_INTERLOCK;
		
		typedef F_INTERLOCK *			F_MUTEX;
		typedef F_INTERLOCK	**			F_MUTEX_p;
		#define F_MUTEX_NULL				NULL

		typedef HANDLE						F_SEM;
		typedef HANDLE *					F_SEM_p;
		#define F_SEM_NULL				NULL

		#define f_stricmp( str1, str2) \
			_stricmp((char *)(str1), (char *)(str2))

		#define f_strnicmp( str1, str2, size) \
			_strnicmp((char *)(str1), (char *)(str2),(size_t)(size))

		#define f_memmove( dest, src, length) \
			memmove((void  *)(dest), (void  *)(src),(size_t)(length))

		#define f_memset( src, chr, size) \
			memset((void  *)(src),(chr),(size_t)(size))

		#define f_memcmp( str1, str2, length) \
			memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))

		#define f_strcat( dest, src) \
			strcat( (char*)(dest), (char*)(src))

		#define f_strchr( str, value) \
			strchr( (char*)str, (int)value)

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
			(char *)strstr( (char*)(str1), (char*)(str2))

		#define f_strncat( str1, str2, n) \
			strncat( (char *)(str1), (char *)(str2), n)

		#define f_strupr( str) \
			_strupr( (char *)(str))

	#endif

	/****************************************************************************
	Desc:		UNIX
	****************************************************************************/
	#if defined( FLM_UNIX)

		#define FSTATIC		static

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

		#define f_stricmp(str1,str2) \
			strcasecmp((char *)(str1),(char *)(str2))

		#define f_strnicmp(str1,str2,size_t) \
			strncasecmp((char *)(str1),(char *)(str2),size_t)

		#define f_memmove( dest, src, len) \
			memmove( (void*)(dest), (void*)(src), len)

		#define f_memset( src, chr, size) \
			memset((void  *)(src),(chr),(size_t)(size))

		#define f_memcmp( str1, str2, length) \
			memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))

		#define f_strcat( dest, src) \
			strcat( (char*)(dest), (char*)(src))

		#define f_strchr( str, value) \
			strchr( (char*)str, (int)value)

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
			(char *)strstr( (char*)(str1), (char*)(str2))

		#define f_strncat( str1, str2, n) \
			strncat( (char *)(str1), (char *)(str2), n)

		#define f_strupr( str) \
			strupr( (char *)(str))

		#define f_va_list				va_list
		#define f_va_start			va_start
		#define f_va_arg				va_arg
		#define f_va_end				va_end

		typedef pthread_mutex_t *	F_MUTEX;
		typedef F_MUTEX *				F_MUTEX_p;
		#define F_MUTEX_NULL			NULL
		
		void f_mutexLock(
			F_MUTEX		hMutex);
	
		void f_mutexUnlock(
			F_MUTEX		hMutex);
		
		typedef struct
		{
			pthread_mutex_t lock;
			pthread_cond_t  cond;
			int             count;
		} sema_t;
	
		typedef sema_t *				F_SEM;
		typedef F_SEM *				F_SEM_p;
		#define F_SEM_NULL			NULL

		typedef int						SOCKET;
		#define INVALID_SOCKET		-1
	
	#endif

	/****************************************************************************
	Desc: Cross-platform inline functions
	****************************************************************************/
	FINLINE void f_memcpy(
		void *			pvDest,
		const void *	pvSrc,
		FLMSIZET			iSize)
	{
		if( iSize == 1)
		{
			*((FLMBYTE *)pvDest) = *((FLMBYTE *)pvSrc);
		}
		else
		{
			(void)memcpy( pvDest, pvSrc, iSize);
		}
	}

	#if defined( __va_copy)
		#define  f_va_copy(to, from) __va_copy(to, from)
	#else
		#define f_va_copy(to, from)  ((to) = (from))
	#endif

	/****************************************************************************
	Desc:		Internal base class
	****************************************************************************/
	class F_OSBase
	{
	public:

		F_OSBase()
		{ 
			m_refCnt = 1;	
		}

		virtual ~F_OSBase()
		{
		}

		FINLINE FLMUINT getRefCount( void)
		{
			return( m_refCnt);
		}

		void * operator new(
			FLMSIZET			uiSize);

	#ifdef FLM_DEBUG
		void * operator new(
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	#endif

		void operator delete(
			void *			ptr);

		void operator delete[](
			void *			ptr);

	#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
		void operator delete(
			void *			ptr,
			const char *,	// file
			int);				// line
	#endif

	#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
		void operator delete[](
			void *			ptr,
			const char *,	// file
			int);				// line
	#endif

		virtual FINLINE FLMINT XFLMAPI AddRef( void)
		{
			return( ++m_refCnt);
		}

		virtual FINLINE FLMINT XFLMAPI Release( void)
		{
			FLMINT		iRefCnt = --m_refCnt;

			if( !iRefCnt)
			{
				delete this;
			}

			return( iRefCnt);
		}

	protected:

		FLMATOMIC		m_refCnt;
	};

	/****************************************************************************
	Desc:		Base class
	****************************************************************************/
	class XF_Base
	{
	public:
	
		XF_Base()
		{
		}
	
		virtual ~XF_Base()
		{
		}
	
		void * operator new(
			FLMSIZET			uiSize);
	
		void * operator new[](
			FLMSIZET			uiSize);
	
	#ifdef FLM_DEBUG
		void * operator new(
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	#endif
	
	#ifdef FLM_DEBUG
		void * operator new[](
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	#endif
	
		void operator delete(
			void *			ptr);
	
		void operator delete[](
			void *			ptr);
	
	#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
		void operator delete(
			void *			ptr,
			const char *	file,
			int				line);
	
		void operator delete[](
			void *			ptr,
			const char *	file,
			int				line);
	#endif
	
	};

	/****************************************************************************
	Desc:	This class is used to do pool memory allocations.
	****************************************************************************/
	class F_Pool : public IF_Pool, public XF_Base
	{
	public:

		typedef struct PoolMemoryBlock
		{
			PoolMemoryBlock *		pPrevBlock;
			FLMUINT					uiBlockSize;
			FLMUINT					uiFreeOffset;
			FLMUINT					uiFreeSize;
		} MBLK;

		typedef struct
		{
			FLMUINT	uiAllocBytes;
			FLMUINT	uiCount;
		} POOL_STATS;

		F_Pool()
		{
			m_uiBytesAllocated = 0;
			m_pLastBlock = NULL;
			m_pPoolStats = NULL;
			m_uiBlockSize = 0;
		}

		~F_Pool();

		FINLINE void poolInit(
			FLMUINT			uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
		}

		void smartPoolInit(
			POOL_STATS *	pPoolStats);

		RCODE poolAlloc(
			FLMUINT			uiSize,
			void **			ppvPtr);

		RCODE poolCalloc(
  			FLMUINT			uiSize,
			void **			ppvPtr);

		void poolFree( void);

		void poolReset(
			void *			pvMark,
			FLMBOOL			bReduceFirstBlock = FALSE);

		FINLINE void * poolMark( void)
		{
			return (void *)(m_pLastBlock
								 ? (FLMBYTE *)m_pLastBlock + m_pLastBlock->uiFreeOffset
								 : NULL);
		}

		FINLINE FLMUINT getBlockSize( void)
		{
			return( m_uiBlockSize);
		}

		FINLINE FLMUINT getBytesAllocated( void)
		{
			return( m_uiBytesAllocated);
		}

	private:

		FINLINE void updateSmartPoolStats( void)
		{
			if (m_uiBytesAllocated)
			{
				if( (m_pPoolStats->uiAllocBytes + m_uiBytesAllocated) >= 0xFFFF0000)
				{
					m_pPoolStats->uiAllocBytes =
						(m_pPoolStats->uiAllocBytes / m_pPoolStats->uiCount) * 100;
					m_pPoolStats->uiCount = 100;
				}
				else
				{
					m_pPoolStats->uiAllocBytes += m_uiBytesAllocated;
					m_pPoolStats->uiCount++;
				}
				m_uiBytesAllocated = 0;
			}
		}

		FINLINE void setInitialSmartPoolBlkSize( void)
		{
			// Determine starting block size:
			// 1) average of bytes allocated / # of frees/resets (average size needed)
			// 2) add 10% - to minimize extra allocs 

			m_uiBlockSize = (m_pPoolStats->uiAllocBytes / m_pPoolStats->uiCount);
			m_uiBlockSize += (m_uiBlockSize / 10);

			if (m_uiBlockSize < 512)
			{
				m_uiBlockSize = 512;
			}
		}

		void freeToMark(
			void *		pvMark);

		PoolMemoryBlock *		m_pLastBlock;
		FLMUINT					m_uiBlockSize;
		FLMUINT					m_uiBytesAllocated;
		POOL_STATS *			m_pPoolStats;
	};

	/****************************************************************************
										CROSS PLATFORM DEFINITIONS
	****************************************************************************/
	#define F_UNREFERENCED_PARM( parm) \
		(void)parm
		
	#define f_min(a, b) \
		((a) < (b) ? (a) : (b))
		
	#define f_max(a, b) \
		((a) < (b) ? (b) : (a))
		
	#define f_swap( a, b, tmp) \
		((tmp) = (a), (a) = (b), (b) = (tmp))
	
	char * f_uwtoa(
		FLMUINT16	value,
		char *		ptr);

	char * f_udtoa(
		FLMUINT		value,
		char *		ptr);

	char * f_wtoa(
		FLMINT16		value,
		char *		ptr);

	char * f_dtoa(
		FLMINT		value,
		char *		ptr);

	char * f_ui64toa(
		FLMUINT64	value,
		char *		ptr);

	char * f_i64toa(
		FLMINT64		value,
		char *		ptr);

	FLMINT f_atoi(
		char *		ptr);

	FLMINT f_atol(
		char *		ptr);

	FLMINT f_atod(
		char *		ptr);

	FLMUINT f_atoud(
		char *		ptr,
		FLMBOOL		bAllowUnprefixedHex = FALSE);

	FLMUINT64 f_atou64(
		char *  		pszBuf);

	FLMUINT f_unilen(
		const FLMUNICODE *	puzStr);

	FLMUNICODE * f_unicpy(
		FLMUNICODE *			puzDestStr,
		const FLMUNICODE *	puzSrcStr);

	FLMUNICODE f_unitolower(
		FLMUNICODE				uChar);

	FLMINT f_unicmp(
		const FLMUNICODE *	puzStr1,
		const FLMUNICODE *	puzStr2);

	FLMINT f_uniicmp(
		const FLMUNICODE *	puzStr1,
		const FLMUNICODE *	puzStr2);

	FLMINT f_uninativecmp(
		const FLMUNICODE *	puzStr1,
		const char *			pszStr2);

	FLMINT f_uninativencmp(
		const FLMUNICODE *	puzStr1,
		const char  *			pszStr2,
		FLMUINT					uiCount);

	/****************************************************************************
									FLAIM's Assert Layer

	This section contains prototypes and macros for FLAIM's assert. This layer
	enables FLAIM to redirect assert calls.
	****************************************************************************/

	#ifdef FLM_DEBUG

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
			extern "C" void EnterDebugger(void);

			#ifdef FLM_DBG_LOG
				#define flmAssert( exp)	\
					(void)( (exp) || (flmDbgLogFlush(), EnterDebugger(), 0))
			#else
				#define flmAssert( exp) \
					(void)( (exp) || ( EnterDebugger(), 0))
			#endif

		#elif defined( FLM_UNIX)
			#ifdef FLM_DBG_LOG
				#define flmAssert( exp) \
					(void)( (exp) || (flmDbgLogFlush(), assert(0), 0))
			#else
				#define flmAssert( exp)	\
					(void)( (exp) || (assert(0), 0))
			#endif

		#else
			#define flmAssert( exp)
		#endif

	#else
		#define flmAssert( exp)
	#endif

	FLMUINT f_breakpoint(
		FLMUINT		uiBreakFlag);

	/****************************************************************************
									Character Value Constants
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

	#define f_stringToAscii( str)

	#define f_toascii( native) \
		(native)

	#define f_tonative( ascii) \
		(ascii)

	#define f_toupper( native) \
		(((native) >= 'a' && (native) <= 'z') \
				? (native) - 'a' + 'A' \
				: (native))

	#define f_tolower( native) \
		(((native) >= 'A' && (native) <= 'Z') \
				? (native) - 'A' + 'a' \
				: (native))

	#define f_islower( native) \
		((native) >= 'a' && (native) <= 'z')

	#ifndef FLM_ASCII_PLATFORM
		#define FLM_ASCII_PLATFORM
	#endif

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

	FINLINE FLMBOOL f_isvowel(
		FLMUNICODE		uChar)
	{
		uChar = f_unitolower( uChar);

		if( uChar == FLM_UNICODE_a ||
			 uChar == FLM_UNICODE_e ||
			 uChar == FLM_UNICODE_i ||
			 uChar == FLM_UNICODE_o ||
			 uChar == FLM_UNICODE_u ||
			 uChar == FLM_UNICODE_y)
		{
			return( TRUE);
		}

		return( FALSE);
	}

	/****************************************************************************
									WORD/BYTE ORDERING MACROS
	****************************************************************************/

	FINLINE FLMUINT32 byteToLong(
		const FLMBYTE *		pucBuf)
	{
		FLMUINT32		ui32Val = 0;

		ui32Val |= ((FLMUINT32)pucBuf[ 0]) << 24;
		ui32Val |= ((FLMUINT32)pucBuf[ 1]) << 16;
		ui32Val |= ((FLMUINT32)pucBuf[ 2]) << 8;
		ui32Val |= ((FLMUINT32)pucBuf[ 3]);

		return( ui32Val);
	}
	
	FINLINE FLMUINT64 byteToLong64(
		const FLMBYTE *		pucBuf)
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

	FLMUINT32 byteToInt( const FLMBYTE * ptr);
	#define  byteToInt(p)  ( \
 		 (FLMUINT16) ( ((((FLMBYTE *)(p))[ 0]) << 8) | (((FLMBYTE *)(p))[ 1]) ) )

	void longToByte( FLMINT32 uiNum, FLMBYTE * ptr);
	#define longToByte( n, p) { \
		FLMUINT32 ui32Temp = (FLMUINT32) (n); FLMBYTE * pTemp = (FLMBYTE *)(p); \
				pTemp[0] = (FLMBYTE) (ui32Temp >> 24); \
				pTemp[1] = (FLMBYTE) (ui32Temp >> 16); \
				pTemp[2] = (FLMBYTE) (ui32Temp >>  8); \
				pTemp[3] = (FLMBYTE) (ui32Temp      ); \
		}

	void long64ToByte( FLMINT64 uiNum, FLMBYTE * ptr);
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

	void intToByte( FLMINT16 uiNum, FLMBYTE * ptr);
	#define intToByte( n, p) { \
		FLMUINT16 ui16Temp = (FLMUINT16) (n); FLMBYTE * pTemp = (FLMBYTE *) (p); \
				pTemp[0] = (FLMBYTE) (ui16Temp >>  8); \
				pTemp[1] = (FLMBYTE) (ui16Temp      ); \
		}

	#ifndef FLM_BIG_ENDIAN

		#if defined( FLM_SPARC)
			#error Wrong endian order selected
		#endif
	
		#define LO(wrd) 	(*(FLMUINT8  *)&wrd)
		#define HI(wrd) 	(*((FLMUINT8 *)&wrd + 1))

		#if defined( FLM_STRICT_ALIGNMENT)

			#define FB2UW( bp) \
				((FLMUINT16)((((FLMUINT16)(((FLMUINT8 *)(bp))[1])) << 8) | \
								 (((FLMUINT16)(((FLMUINT8 *)(bp))[0])))))
	
			#define FB2UD( bp) \
				((FLMUINT32)((((FLMUINT32)(((FLMUINT8 *)(bp))[3]))<< 24) | \
								 (((FLMUINT32)(((FLMUINT8 *)(bp))[2]))<< 16) | \
								 (((FLMUINT32)(((FLMUINT8 *)(bp))[1]))<< 8)  | \
								 (((FLMUINT32)(((FLMUINT8 *)(bp))[0])))))
	
			#define FB2U64( bp)	\
				((FLMUINT64)((((FLMUINT64)(((FLMUINT8 *)(bp))[7])) << 56) | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[6])) << 48) | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[5])) << 40) | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[4])) << 32) | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[3])) << 24) | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[2])) << 16) | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[1]))<< 8)   | \
								 (((FLMUINT64)(((FLMUINT8 *)(bp))[0])))))
	
			#define UW2FBA( uw, bp)	\
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)(uw)), \
				 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)((((uw) & 0xff00) >> 8))))
	
			#define UD2FBA( udw, bp) \
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((udw) & 0xff)), \
				 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((udw) & 0xff00) >> 8)), \
				 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((udw) & 0xff0000) >> 16)), \
				 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((udw) & 0xff000000) >> 24)))
	
			#define U642FBA( u64, bp) \
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((u64) & 0xff)), \
				 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((u64) & 0xff00) >> 8)), \
				 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((u64) & 0xff0000) >> 16)), \
				 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((u64) & 0xff000000) >> 24)), \
				 ((FLMUINT8 *)(bp))[4] = ((FLMUINT8)(((u64) & 0xff00000000) >> 32)), \
				 ((FLMUINT8 *)(bp))[5] = ((FLMUINT8)(((u64) & 0xff0000000000) >> 40)), \
				 ((FLMUINT8 *)(bp))[6] = ((FLMUINT8)(((u64) & 0xff000000000000) >> 48)), \
				 ((FLMUINT8 *)(bp))[7] = ((FLMUINT8)(((u64) & 0xff00000000000000) >> 56)))
		#else
			#define FB2UW( fbp) \
				(*((FLMUINT16 *)(fbp)))
				
			#define FB2UD( fbp) \
				(*((FLMUINT32 *)(fbp)))
				
			#define FB2U64( fbp) \
				(*((FLMUINT64 *)(fbp)))
				
			#define UW2FBA( uw, fbp) \
				(*((FLMUINT16 *)(fbp)) = ((FLMUINT16) (uw)))
				
			#define UD2FBA( uw, fbp) \
				(*((FLMUINT32 *)(fbp)) = ((FLMUINT32) (uw)))
				
			#define U642FBA( uw, fbp) \
				(*((FLMUINT64 *)(fbp)) = ((FLMUINT64) (uw)))
		#endif

	#else

		#if defined( __i386__)
			#error Wrong endian order selected
		#endif

		#define LO( wrd) \
			(*((FLMUINT8 *)&wrd + 1))
			
		#define HI( wrd) \
			(*(FLMUINT8  *)&wrd)

		#define FB2UW( bp) \
			((FLMUINT16)((((FLMUINT16)(((FLMUINT8 *)(bp))[1])) << 8) | \
							 (((FLMUINT16)(((FLMUINT8 *)(bp))[0])))))

		#define FB2UD( bp) \
			((FLMUINT32)((((FLMUINT32)(((FLMUINT8 *)(bp))[3])) << 24) | \
							 (((FLMUINT32)(((FLMUINT8 *)(bp))[2])) << 16) | \
							 (((FLMUINT32)(((FLMUINT8 *)(bp))[1])) << 8)  | \
							 (((FLMUINT32)(((FLMUINT8 *)(bp))[0])))))

		#define FB2U64( bp) \
			((FLMUINT64)((((FLMUINT64)(((FLMUINT8 *)(bp))[7])) << 56) | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[6])) << 48) | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[5])) << 40) | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[4])) << 32) | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[3])) << 24) | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[2])) << 16) | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[1])) << 8)  | \
							 (((FLMUINT64)(((FLMUINT8 *)(bp))[0])))))

		#define UW2FBA( uw, bp) \
			(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)(uw)), \
			 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)((((uw) & 0xff00) >> 8))))

		#define UD2FBA( udw, bp) \
			(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((udw) & 0xff)), \
			 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((udw) & 0xff00) >> 8)), \
			 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((udw) & 0xff0000) >> 16)), \
			 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((udw) & 0xff000000) >> 24)))

		#ifdef FLM_UNIX
			#define U642FBA( u64, bp) \
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((u64) & 0xffULL)), \
				 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((u64) & 0xff00ULL) >> 8)), \
				 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((u64) & 0xff0000ULL) >> 16)), \
				 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((u64) & 0xff000000ULL) >> 24)), \
				 ((FLMUINT8 *)(bp))[4] = ((FLMUINT8)(((u64) & 0xff00000000ULL) >> 32)), \
				 ((FLMUINT8 *)(bp))[5] = ((FLMUINT8)(((u64) & 0xff0000000000ULL) >> 40)), \
				 ((FLMUINT8 *)(bp))[6] = ((FLMUINT8)(((u64) & 0xff000000000000ULL) >> 48)), \
				 ((FLMUINT8 *)(bp))[7] = ((FLMUINT8)(((u64) & 0xff00000000000000ULL) >> 56)))
		#else
			#define	U642FBA( u64, bp) \
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((u64) & 0xff)), \
				 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((u64) & 0xff00) >> 8)), \
				 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((u64) & 0xff0000) >> 16)), \
				 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((u64) & 0xff000000) >> 24)), \
				 ((FLMUINT8 *)(bp))[4] = ((FLMUINT8)(((u64) & 0xff00000000) >> 32)), \
				 ((FLMUINT8 *)(bp))[5] = ((FLMUINT8)(((u64) & 0xff0000000000) >> 40)), \
				 ((FLMUINT8 *)(bp))[6] = ((FLMUINT8)(((u64) & 0xff000000000000) >> 48)), \
				 ((FLMUINT8 *)(bp))[7] = ((FLMUINT8)(((u64) & 0xff00000000000000) >> 56)))
		#endif
	#endif

	/****************************************************************************
									File Path Functions & Macros
	****************************************************************************/

	// This defines the maximum file size we can support for ANY
	// platform, ANY file type.  It is not 4Gb because of a bug in direct IO
	// on Netware.  The limitation is that in direct IO mode (on the legacy file
	// system) we are not allowed room for the last block.  If the block
	// size were 64K for example, direct IO only lets us expand to a size of
	// 0xFFFF0000.  Since we can't anticipate what the block size will be,
	// we have to set a maximum that accounts for the maximum block size we
	// may ever see.  At this point, we are assuming it won't ever be more
	// than 256K on legacy file systems.  Thus, our limit of 0xFFFC0000.
	// (See define of F_MAXIMUM_FILE_SIZE in xflaim.h)

	#if defined( FLM_WIN) || defined( FLM_NLM)
		#define FWSLASH     		'/'
		#define SLASH       		'\\'
		#define SSLASH      		"\\"
		#define COLON       		':'
		#define PERIOD      		'.'
		#define PARENT_DIR  		".."
		#define CURRENT_DIR 		"."
	#else
		#ifndef FWSLASH
			#define FWSLASH 		'/'
		#endif

		#ifndef SLASH
			#define SLASH  		'/'
		#endif

		#ifndef SSLASH
			#define SSLASH			"/"
		#endif

		#ifndef COLON
			#define COLON  		':'
		#endif

		#ifndef PERIOD
			#define PERIOD 		'.'
		#endif

		#ifndef PARENT_DIR
			#define PARENT_DIR 	".."
		#endif

		#ifndef CURRENT_DIR
			#define CURRENT_DIR 	"."
		#endif
	#endif

	/****************************************************************************
									CPU Release Functions
	****************************************************************************/

	#ifdef FLM_NLM
		#define f_yieldCPU() \
			pthread_yield()
	#else
		#define f_yieldCPU()
	#endif

	void f_sleep(
		FLMUINT	uiMilliseconds);

	#ifdef FLM_WIN
		#define f_sleep( uiMilliseconds) \
			Sleep( (DWORD)uiMilliseconds)
	#endif

	/****************************************************************************
										Random Numbers
	****************************************************************************/

	#define MAX_RANDOM  2147483646L

	class F_RandomGenerator : public XF_RefCount, public XF_Base
	{
	public:

		void randomize( void);

		void randomSetSeed(
			FLMINT32		i32seed);

		FLMINT32	randomLong( void);

		FLMINT32 randomChoice(
			FLMINT32 	lo,
			FLMINT32 	hi);

		FLMINT randomTruth(
			FLMINT		iPercentageTrue);

		FLMINT getSeed( void)
		{
			return( m_i32Seed);
		}

	private:

		FLMINT32			m_i32Seed;
	};

	/****************************************************************************
									Time, date, timestamp functions
	****************************************************************************/

	typedef struct
	{
		FLMUINT16	year;
		FLMBYTE		month;
		FLMBYTE		day;
		FLMBYTE		hour;
		FLMBYTE		minute;
		FLMBYTE		second;
		FLMBYTE		hundredth;
	} F_TMSTAMP;

	#define f_timeIsLeapYear(year) \
		((((year) & 0x03) == 0) && (((year) % 100) != 0) || (((year) % 400) == 0))

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
		FLMUINT			flag);

	#if defined( FLM_UNIX)
		unsigned f_timeGetMilliTime();
	#endif

	FINLINE FLMUINT flmLocalToUTC(
		FLMUINT	uiSeconds)
	{
		return( uiSeconds + f_timeGetLocalOffset());
	}

	/**********************************************************************
	Desc: Atomic Increment, Decrement, Exchange
	Note:	Some of this code is derived from the Ximian source code contained
			in that Mono project's atomic.h file. 
	**********************************************************************/
	#ifndef FLM_HAVE_ATOMICS
		#define FLM_HAVE_ATOMICS
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
	FINLINE FLMINT32 _flmAtomicInc(
		FLMATOMIC *			piTarget)
	{
		#if defined( FLM_NLM)
		{
			return( (FLMINT32)nlm_AtomicIncrement( (volatile LONG *)piTarget));
		}
		#elif defined( FLM_WIN)
		{
			return( (FLMINT32)InterlockedIncrement( (volatile LONG *)piTarget));
		}
		#elif defined( FLM_AIX)
		{
			return( (FLMINT32)aix_atomic_add( piTarget, 1));
		}
		#elif defined( FLM_OSX)
		{
			return( (FLMINT32)OSAtomicIncrement32( (int32_t *)piTarget));
		}
		#elif defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
		{
			return( sparc_atomic_add_32( piTarget, 1));
		}
		#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
		{
			FLMINT32 			i32Tmp;
			
			__asm__ __volatile__ (
							"lock;"
							"xaddl %0, %1"
								: "=r" (i32Tmp), "=m" (*piTarget)
								: "0" (1), "m" (*piTarget));
		
			return( i32Tmp + 1);
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
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	FINLINE FLMINT32 _flmAtomicDec(
		FLMATOMIC *			piTarget)
	{
		#if defined( FLM_NLM)
		{
			return( (FLMINT32)nlm_AtomicDecrement( (volatile LONG *)piTarget));
		}
		#elif defined( FLM_WIN)
		{
			return( (FLMINT32)InterlockedDecrement( (volatile LONG *)piTarget));
		}
		#elif defined( FLM_AIX)
		{
			return( (FLMINT32)aix_atomic_add( piTarget, -1));
		}
		#elif defined( FLM_OSX)
		{
			return( (FLMINT32)OSAtomicDecrement32( (int32_t *)piTarget));
		}
		#elif defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
		{
			return( sparc_atomic_add_32( piTarget, -1));
		}
		#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
		{
			FLMINT32				i32Tmp;
			
			__asm__ __volatile__ (
							"lock;" 
							"xaddl %0, %1"
								: "=r" (i32Tmp), "=m" (*piTarget)
								: "0" (-1), "m" (*piTarget));
		
			return( i32Tmp - 1);
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
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	FINLINE FLMINT32 _flmAtomicExchange(
		FLMATOMIC *			piTarget,
		FLMINT32				i32NewVal)
	{
		#if defined( FLM_NLM)
		{
			return( (FLMINT32)nlm_AtomicExchange( 
				(volatile LONG *)piTarget, i32NewVal));
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
		#elif defined( FLM_OSX)
		{
			int32_t		iOldVal;

			for( ;;)
			{
				iOldVal = (int32_t)*piTarget;

				if( OSAtomicCompareAndSwap32( iOldVal, i32NewVal, 
						(int32_t *)piTarget))
				{
					break;
				}
			}
			
			return( (FLMINT32)iOldVal);
		}
		#elif defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
		{
			return( sparc_atomic_xchg_32( piTarget, i32NewVal));
		}
		#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
		{
			FLMINT32 			i32OldVal;
			
			__asm__ __volatile__ (
							"1:	lock;"
							"		cmpxchgl %2, %0;"
							"		jne 1b"
								: "=m" (*piTarget), "=a" (i32OldVal)
								: "r" (i32NewVal), "m" (*piTarget), "a" (*piTarget));
		
			return( i32OldVal);
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
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	FINLINE FLMINT32 flmAtomicInc(
		FLMATOMIC *		piTarget,
		F_MUTEX			hMutex = F_MUTEX_NULL,
		FLMBOOL			bMutexAlreadyLocked = FALSE)
	{
		#ifdef FLM_HAVE_ATOMICS
		{
			F_UNREFERENCED_PARM( bMutexAlreadyLocked);
			F_UNREFERENCED_PARM( hMutex);
			
			return( _flmAtomicInc( piTarget));
		}
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
		{
			F_UNREFERENCED_PARM( bMutexAlreadyLocked);
			F_UNREFERENCED_PARM( hMutex);
			
			return( _flmAtomicDec( piTarget));
		}
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
		{
			F_UNREFERENCED_PARM( bMutexAlreadyLocked);
			F_UNREFERENCED_PARM( hMutex);
			
			return( _flmAtomicExchange( piTarget, i32NewVal));
		}
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
	Desc: Mutex and semaphore routines
	****************************************************************************/
	#ifdef FLM_NLM
		FINLINE RCODE f_mutexCreate(
			F_MUTEX *	phMutex)
		{
			if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"NOVDB")) == F_MUTEX_NULL)
			{
				return RC_SET( NE_XFLM_MEM);
			}
	
			return NE_XFLM_OK;
		}
	
		FINLINE void f_mutexDestroy(
			F_MUTEX *	phMutex)
		{
			if (*phMutex != F_MUTEX_NULL)
			{
				if( kMutexFree( (MUTEX)(*phMutex)))
				{
					flmAssert( 0);
				}
				
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
	
		FINLINE void f_assertMutexLocked(
			F_MUTEX)
		{
		}

		typedef SEMAPHORE				F_SEM;
		typedef SEMAPHORE *			F_SEM_p;
		#define F_SEM_NULL			0

		FINLINE RCODE f_semCreate(
			F_SEM *		phSem)
		{
			if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"NOVDB", 0)) == F_SEM_NULL)
			{
				return RC_SET( NE_XFLM_MEM);
			}
	
			return NE_XFLM_OK;
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
			RCODE			rc = NE_XFLM_OK;
	
			if( uiTimeout == F_SEM_WAITFOREVER)
			{
				if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
				{
					rc = RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE);
				}
			}
			else
			{
				if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
				{
					rc = RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE);
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
	
		RCODE f_mutexCreate(
			F_MUTEX *	phMutex);
	
		void f_mutexDestroy(
			F_MUTEX *	phMutex);
			
		FINLINE void f_mutexLock(
			F_MUTEX		hMutex)
		{
			while( flmAtomicExchange( 
				&(((F_INTERLOCK *)hMutex)->locked), 1) != 0)
			{
		#ifdef FLM_DEBUG
				flmAtomicInc( &(((F_INTERLOCK *)hMutex)->waitCount));
		#endif
				Sleep( 0);
			}
	
		#ifdef FLM_DEBUG
			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == 0);
			((F_INTERLOCK *)hMutex)->uiThreadId = _threadid;
			flmAtomicInc( &(((F_INTERLOCK *)hMutex)->lockedCount));
		#endif
		}
		
		FINLINE void f_mutexUnlock(
			F_MUTEX		hMutex)
		{
			flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
		#ifdef FLM_DEBUG
			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
			((F_INTERLOCK *)hMutex)->uiThreadId = 0;
		#endif
			flmAtomicExchange( &(((F_INTERLOCK *)hMutex)->locked), 0);
		}
	
		FINLINE void f_assertMutexLocked(
			F_MUTEX		hMutex)
		{
		#ifdef FLM_DEBUG
			flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
		#else
			F_UNREFERENCED_PARM( hMutex);
		#endif
		}
		
		FINLINE RCODE f_semCreate(
			F_SEM *		phSem)
		{
			if( (*phSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL,
				0, 10000, NULL )) == NULL)
			{
				return( RC_SET( NE_XFLM_COULD_NOT_CREATE_SEMAPHORE));
			}
	
			return NE_XFLM_OK;
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
				return( NE_XFLM_OK);
			}
			else
			{
				return( RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE));
			}
		}
	
		FINLINE void f_semSignal(
			F_SEM			hSem)
		{
			(void)ReleaseSemaphore( hSem, 1, NULL);
		}
	#elif defined( FLM_UNIX)
		RCODE f_mutexCreate(
			F_MUTEX *	phMutex);
	
		void f_mutexDestroy(
			F_MUTEX *	phMutex);
		
		FINLINE void f_mutexLock(
			F_MUTEX		hMutex)
		{
			(void)pthread_mutex_lock( hMutex);
		}
	
		FINLINE void f_mutexUnlock(
			F_MUTEX		hMutex)
		{
			(void)pthread_mutex_unlock( hMutex);
		}
	
		FINLINE void f_assertMutexLocked(
			F_MUTEX)
		{
		}

		int sema_signal(
			sema_t *			sem);

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
			(void)sema_signal( hSem);
		}

	#endif

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
	Desc:
	****************************************************************************/
	class F_IStream : public IF_IStream, public XF_Base
	{
	public:

		F_IStream();

		virtual ~F_IStream();

		void lockModule( void);

	private:

		FLMBOOL		m_bLockedModule;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_OStream : public IF_OStream, public XF_Base
	{
	public:

		F_OStream();

		virtual ~F_OStream();

		void lockModule( void);

	private:

		FLMBOOL		m_bLockedModule;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_PosIStream : public IF_PosIStream, public XF_Base
	{
	public:

		F_PosIStream();

		virtual ~F_PosIStream();

		void lockModule( void);

	private:

		FLMBOOL		m_bLockedModule;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferIStream : public F_PosIStream
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

		RCODE XFLMAPI open(
			const FLMBYTE *	pucBuffer,
			FLMUINT				uiLength,
			FLMBYTE **			ppucAllocatedBuffer = NULL);

		FINLINE FLMUINT64 XFLMAPI totalSize( void)
		{
			flmAssert( m_bIsOpen);
			return( m_uiBufferLen);
		}

		FINLINE FLMUINT64 XFLMAPI remainingSize( void)
		{
			flmAssert( m_bIsOpen);
			return( m_uiBufferLen - m_uiOffset);
		}

		RCODE XFLMAPI close( void);

		FINLINE RCODE XFLMAPI positionTo(
			FLMUINT64		ui64Position)
		{
			flmAssert( m_bIsOpen);

			if( ui64Position < m_uiBufferLen)
			{
				m_uiOffset = (FLMUINT)ui64Position;
			}
			else
			{
				m_uiOffset = m_uiBufferLen;
			}

			return( NE_XFLM_OK);
		}

		FINLINE FLMUINT64 XFLMAPI getCurrPosition( void)
		{
			flmAssert( m_bIsOpen);
			return( m_uiOffset);
		}

		RCODE XFLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
			
		FINLINE const FLMBYTE * getBuffer( void)
		{
			flmAssert( m_bIsOpen);
			return( m_pucBuffer);
		}
		
		FINLINE const FLMBYTE * getBufferAtCurrentOffset( void)
		{
			flmAssert( m_bIsOpen);
			return( m_pucBuffer ? &m_pucBuffer[ m_uiOffset] : NULL);
		}
		
		FINLINE void truncate(
			FLMUINT		uiOffset)
		{
			flmAssert( m_bIsOpen);
			flmAssert( uiOffset >= m_uiOffset);
			flmAssert( uiOffset <= m_uiBufferLen);
			
			m_uiBufferLen = uiOffset;
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

		RCODE XFLMAPI open(
			const char *	pszPath);

		RCODE XFLMAPI close( void);

		RCODE XFLMAPI positionTo(
			FLMUINT64		ui64Position);

		FLMUINT64 XFLMAPI totalSize( void);

		FLMUINT64 XFLMAPI remainingSize( void);

		FLMUINT64 XFLMAPI getCurrPosition( void);

		RCODE XFLMAPI read(
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

		RCODE XFLMAPI open(
			IF_IStream *		pIStream,
			FLMUINT				uiBufferSize);

		RCODE XFLMAPI read(
			void *				pvBuffer,
			FLMUINT				uiBytesToRead,
			FLMUINT *			puiBytesRead);

		RCODE XFLMAPI close( void);

		FINLINE FLMUINT64 XFLMAPI totalSize( void)
		{
			if (!m_pIStream)
			{
				flmAssert( 0);
				return( 0);
			}

			return( m_uiBytesAvail);
		}

		FINLINE FLMUINT64 XFLMAPI remainingSize( void)
		{
			if( !m_pIStream)
			{
				flmAssert( 0);
				return( 0);
			}

			return( m_uiBytesAvail - m_uiBufferOffset);
		}

		FINLINE RCODE XFLMAPI positionTo(
			FLMUINT64		ui64Position)
		{
			if( !m_pIStream)
			{
				flmAssert( 0);
				return( RC_SET( NE_XFLM_ILLEGAL_OP));
			}

			if( ui64Position < m_uiBytesAvail)
			{
				m_uiBufferOffset = (FLMUINT)ui64Position;
			}
			else
			{
				m_uiBufferOffset = m_uiBytesAvail;
			}

			return( NE_XFLM_OK);
		}

		FINLINE FLMUINT64 XFLMAPI getCurrPosition( void)
		{
			if( !m_pIStream)
			{
				flmAssert( 0);
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

		RCODE XFLMAPI open(
			IF_OStream *	pOStream,
			FLMUINT			uiBufferSize);

		RCODE XFLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE XFLMAPI close( void);

		RCODE XFLMAPI flush( void);

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

		RCODE XFLMAPI open(
			const char *	pszFilePath,
			FLMBOOL			bTruncateIfExists);

		RCODE XFLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE XFLMAPI close( void);

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

		RCODE XFLMAPI open(
			const char *	pszDirectory,
			const char *	pszBaseName);

		RCODE XFLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

		RCODE XFLMAPI close( void);

	private:

		RCODE rollToNextFile( void);

		IF_IStream *		m_pIStream;
		FLMBOOL				m_bOpen;
		FLMBOOL				m_bEndOfStream;
		FLMUINT				m_uiFileNum;
		FLMUINT64			m_ui64FileOffset;
		FLMBYTE 				m_szDirectory[ F_PATH_MAX_SIZE + 1];
		FLMBYTE 				m_szBaseName[ F_PATH_MAX_SIZE + 1];
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

		RCODE XFLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE XFLMAPI close( void);

	private:

		RCODE rollToNextFile( void);

		RCODE processDirectory(
			const char *	pszDirectory,
			const char *	pszBaseName,
			FLMBOOL			bOkToDelete);

		F_OStream *		m_pOStream;
		FLMBOOL			m_bOpen;
		FLMUINT			m_uiFileNum;
		FLMUINT64		m_ui64MaxFileSize;
		FLMUINT64		m_ui64FileOffset;
		FLMBYTE 			m_szDirectory[ F_PATH_MAX_SIZE + 1];
		FLMBYTE 			m_szBaseName[ F_PATH_MAX_SIZE + 1];
		
		friend class F_DbSystem;
	};

	/****************************************************************************
	Desc:	Decodes an ASCII base64 stream to binary
	****************************************************************************/
	class F_Base64DecoderIStream : public F_IStream
	{
	public:

		F_Base64DecoderIStream()
		{
			m_pIStream = NULL;
			m_uiBufOffset = 0;
			m_uiAvailBytes = 0;
		}

		~F_Base64DecoderIStream()
		{
			close();
		}

		RCODE XFLMAPI open(
			IF_IStream *	pIStream);
		
		RCODE XFLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
			
		FINLINE RCODE XFLMAPI close( void)
		{
			RCODE		rc = NE_XFLM_OK;
			
			if( m_pIStream)
			{
				if( m_pIStream->getRefCount() == 1)
				{
					rc = m_pIStream->close();
				}

				m_pIStream->Release();
				m_pIStream = NULL;
			}
			
			m_uiAvailBytes = 0;
			m_uiBufOffset = 0;
			
			return( rc);
		}
		
	private:

		IF_IStream *		m_pIStream;
		FLMUINT				m_uiBufOffset;
		FLMUINT				m_uiAvailBytes;
		FLMBYTE				m_ucBuffer[ 8];
		static FLMBYTE		m_ucDecodeTable[ 256];
	};

	/****************************************************************************
	Desc:	Encodes a binary input stream into ASCII base64.
	****************************************************************************/
	class F_Base64EncoderIStream : public F_IStream
	{
	public:

		F_Base64EncoderIStream()
		{
			m_pIStream = NULL;
		}

		~F_Base64EncoderIStream()
		{
			close();
		}

		RCODE XFLMAPI open(
			IF_IStream *	pIStream,
			FLMBOOL			bLineBreaks);
		
		RCODE XFLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

		FINLINE RCODE XFLMAPI close( void)
		{
			RCODE		rc = NE_XFLM_OK;
			
			if( m_pIStream)
			{
				if( m_pIStream->getRefCount() == 1)
				{
					rc = m_pIStream->close();
				}

				m_pIStream->Release();
				m_pIStream = NULL;
			}
			
			return( rc);
		}
		
	private:

		IF_IStream *		m_pIStream;
		FLMBOOL				m_bInputExhausted;
		FLMBOOL				m_bLineBreaks;
		FLMBOOL				m_bPriorLineEnd;
		FLMUINT				m_uiBase64Count;
		FLMUINT				m_uiBufOffset;
		FLMUINT				m_uiAvailBytes;
		FLMBYTE 				m_ucBuffer[ 8];
		static FLMBYTE		m_ucEncodeTable[ 64];
	};

	typedef struct LZWODictItem
	{
		LZWODictItem *	pNext;
		FLMUINT16		ui16Code;
		FLMUINT16		ui16ParentCode;
		FLMBYTE			ucChar;
	} LZWODictItem;

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_CompressingOStream : public F_OStream
	{
	public:

		F_CompressingOStream()
		{
			m_pOStream = NULL;
			m_ppHashTbl = NULL;
			m_pool.poolInit( 64 * 1024);
		}

		virtual ~F_CompressingOStream()
		{
			close();
		}

		RCODE XFLMAPI open(
			IF_OStream *	pOStream);

		RCODE XFLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE XFLMAPI close( void);

	private:

		FINLINE FLMUINT getHashBucket(
			FLMUINT16	ui16CurrentCode,
			FLMBYTE		ucChar)
		{
			return( ((((FLMUINT)ui16CurrentCode) << 8) | 
				((FLMUINT)ucChar)) % m_uiHashTblSize);
		}

		LZWODictItem * findDictEntry( 
			FLMUINT16		ui16CurrentCode,
			FLMBYTE			ucChar);

		IF_OStream *		m_pOStream;
		LZWODictItem **	m_ppHashTbl;
		FLMUINT				m_uiHashTblSize;
		FLMUINT				m_uiLastRatio;
		FLMUINT				m_uiBestRatio;
		FLMUINT				m_uiCurrentBytesIn;
		FLMUINT				m_uiTotalBytesIn;
		FLMUINT				m_uiCurrentBytesOut;
		FLMUINT				m_uiTotalBytesOut;
		FLMBOOL				m_bStopCompression;
		FLMUINT16			m_ui16CurrentCode;
		FLMUINT16			m_ui16FreeCode;
		F_Pool				m_pool;
	};

	typedef struct LZWIDictItem
	{
		LZWODictItem *	pNext;
		FLMUINT16		ui16ParentCode;
		FLMBYTE			ucChar;
	} LZWIDictItem;

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_UncompressingIStream : public F_IStream
	{
	public:

		F_UncompressingIStream()
		{
			m_pIStream = NULL;
			m_pDict = NULL;
			m_pucDecodeBuffer = NULL;
		}

		virtual ~F_UncompressingIStream()
		{
			close();
		}

		RCODE XFLMAPI open(
			IF_IStream *	pIStream);

		RCODE XFLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

		RCODE XFLMAPI close( void);
		
	private:

		RCODE readCode(
			FLMUINT16 *		pui16Code);

		RCODE decodeToBuffer(
			FLMUINT16		ui16Code);

		IF_IStream *		m_pIStream;
		LZWIDictItem *		m_pDict;
		FLMBYTE *			m_pucDecodeBuffer;
		FLMUINT				m_uiDecodeBufferSize;
		FLMUINT				m_uiDecodeBufferOffset;
		FLMUINT16			m_ui16FreeCode;
		FLMUINT16			m_ui16LastCode;
		FLMBOOL				m_bStopCompression;
		FLMBOOL				m_bEndOfStream;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class	F_TCPStream : public F_IStream, public F_OStream
	{
	public:
	
			F_TCPStream( void);
			
			virtual ~F_TCPStream( void);
	
			RCODE openConnection(
				const char *	pucHostAddress,
				FLMUINT			uiPort,
				FLMUINT			uiConnectTimeout	= 3,
				FLMUINT			uiDataTimeout = 15);
	
			RCODE XFLMAPI read(
				void *			pvBuffer,
				FLMUINT			uiBytesToRead,
				FLMUINT *		puiBytesRead);
				
			RCODE XFLMAPI write(
				const void *	pvBuffer,
				FLMUINT			uiBytesToWrite,
				FLMUINT *		puiBytesWritten);
			
			FINLINE RCODE socketPeekWrite(
				FLMINT		iTimeOut)
			{
				return( socketPeek( iTimeOut, FALSE));
			}
	
			FINLINE RCODE socketPeekRead( 
				FLMINT		iTimeOut)
			{
				return( socketPeek( iTimeOut, TRUE));
			};
	
			FINLINE const char * getName( void)
			{
				getLocalInfo();
				return( (const char *)m_pszName);
			};
	
			FINLINE const char * getAddr( void)
			{
				getLocalInfo();
				return( (const char *)m_pszIp);
			};
	
			FINLINE const char * getPeerName( void)
			{
				getRemoteInfo();
				return( (const char *)m_pszPeerName);
			};
	
			FINLINE const char * getPeerAddr( void)
			{
				getRemoteInfo();
				return( (const char *)m_pszPeerIp);
			};
	
			RCODE readNoWait(
				FLMBYTE *		pucBuffer,
				FLMUINT			uiCount,
				FLMUINT *		puiReadRead);
	
			RCODE readAll(
				FLMBYTE *		pucBuffer,
				FLMUINT			uiCount,
				FLMUINT *		puiBytesRead);
	
			RCODE	setTcpDelay(
				FLMBOOL			bOn);
				
			RCODE XFLMAPI close( void);
	
	private:
	
			RCODE getLocalInfo( void);
			
			RCODE getRemoteInfo( void);
	
			RCODE socketPeek(
				FLMINT			iTimoutVal,
				FLMBOOL			bPeekRead);
	
	#ifndef FLM_UNIX
			WSADATA			m_wsaData;
	#endif
			FLMBOOL			m_bInitialized;
			SOCKET			m_iSocket;
			FLMUINT			m_uiIOTimeout;
			FLMBOOL			m_bConnected;
			char				m_pszIp[ 256];
			char				m_pszName[ 256];
			char				m_pszPeerIp[ 256];
			char				m_pszPeerName[ 256];
			unsigned long	m_ulRemoteAddr;
	};
	
	/****************************************************************************
												Misc.
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

	FINLINE FLMBOOL f_isHexChar(
		FLMUNICODE		uChar)
	{
		if( uChar > 127)
		{
			return( FALSE);
		}

		return( f_isHexChar( f_tonative( (FLMBYTE)uChar)));
	}

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

	FINLINE FLMBYTE f_getHexVal(
		FLMUNICODE	uChar)
	{
		return( f_getHexVal( f_tonative( (FLMBYTE)uChar)));
	}

	FINLINE FLMBOOL f_isValidHexNum(
		const FLMBYTE *	pszString)
	{
		if( *pszString == 0)
		{
			return( FALSE);
		}

		while( *pszString)
		{
			if( !f_isHexChar( *pszString))
			{
				return( TRUE);
			}

			pszString++;
		}

		return( TRUE);
	}

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
			return( f_getNLMHandle());
		}
		
	#else
		#error "Unsupported Platform"
	#endif

	/****************************************************************************
											 f_sprintf
	****************************************************************************/

	typedef struct
	{
		FLMBYTE *	pszDestStr;
	} F_SPRINTF_INFO;

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

	void flmSprintfProcessFieldInfo(
		FLMBYTE **			ppszFormat,
		FLMUINT *			puiWidth,
		FLMUINT *			puiPrecision,
		FLMUINT *			puiFlags,
		f_va_list *			args);

	void flmSprintfStringFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void flmSprintfCharFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void flmSprintfErrorFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void flmSprintfNotHandledFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void flmSprintfNumberFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	FLMINT f_vsprintf(
		char *			pszDestStr,
		const char *	pszFormat,
		f_va_list *		args);

	FLMINT f_sprintf(
		char *			pszDestStr,
		const char *	pszFormat,
		...);

	/****************************************************************************
											 Quick Sort
	****************************************************************************/
	
	typedef FLMINT (* F_SORT_COMPARE_FUNC)(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2);

	typedef void (* F_SORT_SWAP_FUNC)(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2);

	FLMINT flmQSortUINTCompare(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2);

	void flmQSortUINTSwap(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2);

	void f_qsort(
		void *					pvBuffer,
		FLMUINT					uiLowerBounds,
		FLMUINT					uiUpperBounds,
		F_SORT_COMPARE_FUNC	fnCompare,
		F_SORT_SWAP_FUNC		fnSwap);

	/****************************************************************************
											 Environment
	****************************************************************************/
	
	void f_getenv(
		const char *			pszKey,
		FLMBYTE *				pszBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiValueLen = NULL);

	/****************************************************************************
										NECESSARY INCLUDE FILES
	****************************************************************************/

	#include "ftkmem.h"
	
#endif	// FTK_H
