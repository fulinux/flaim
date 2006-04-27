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

	#ifndef ALLOCATE_SYS_DATA
		extern IF_FileSystem *		gv_pFileSystem;
		extern IF_ThreadMgr *		gv_pThreadMgr;
	#else
		IF_FileSystem *				gv_pFileSystem;
		IF_ThreadMgr *					gv_pThreadMgr;
	#endif
	
	/****************************************************************************
	Desc:	Errors
	****************************************************************************/
	#ifdef FLM_DEBUG
		RCODE	f_makeErr(
			RCODE				rc,
			const char *	pszFile,
			int				iLine,
			FLMBOOL			bAssert);
			
		#define RC_SET( rc) \
			f_makeErr( rc, __FILE__, __LINE__, FALSE)
			
		#define RC_SET_AND_ASSERT( rc) \
			f_makeErr( rc, __FILE__, __LINE__, TRUE)
			
		#define RC_UNEXPECTED_ASSERT( rc) \
			f_makeErr( rc, __FILE__, __LINE__, TRUE)
	#else
		#define RC_SET( rc)							(rc)
		#define RC_SET_AND_ASSERT( rc)			(rc)
		#define RC_UNEXPECTED_ASSERT( rc)
	#endif

	RCODE MapPlatformError(
		FLMINT		iError,
		RCODE			defaultRc);

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

		// typedef unsigned long f_va_list;
		
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
	
		extern "C" FLMUINT f_getNLMHandle( void);
		
		extern "C" RCODE f_netwareStartup( void);

		extern "C" void f_netwareShutdown( void);
			
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
		#pragma pack( pop, enter_windows)
		
		// Conversion from XXX to YYY, possible loss of data
		#pragma warning( disable : 4244) 
	
		// Local variable XXX may be used without having been initialized
		#pragma warning( disable : 4701)
	
		// Function XXX not inlined
		#pragma warning( disable : 4710) 
		
		#define FSTATIC			static

		#define ENDLINE			ENDLINE_CRLF
		#define f_va_start		va_start
		#define f_va_arg			va_arg
		#define f_va_end			va_end
		
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

		#define f_va_start			va_start
		#define f_va_arg				va_arg
		#define f_va_end				va_end

		typedef int						SOCKET;
		#define INVALID_SOCKET		-1
	
	#endif

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE void * FLMAPI f_memcpy(
		void *			pvDest,
		const void *	pvSrc,
		FLMSIZET			iSize)
	{
		if( iSize == 1)
		{
			*((FLMBYTE *)pvDest) = *((FLMBYTE *)pvSrc);
			return( pvDest);
		}
		else
		{
			return( memcpy( pvDest, pvSrc, iSize));
		}
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE void * FLMAPI f_memmove(
		void *			pvDest,
		const void *	pvSrc,
		FLMSIZET			uiLength)
	{
		return( memmove( pvDest, pvSrc, uiLength));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE void * FLMAPI f_memset(
		void *				pvDest,
		unsigned char		ucByte,
		FLMSIZET				uiLength)
	{
		return( memset( pvDest, ucByte, uiLength));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE FLMINT FLMAPI f_memcmp(
		const void *		pvMem1,
		const void *		pvMem2,
		FLMSIZET				uiLength)
	{
		return( memcmp( pvMem1, pvMem2, uiLength));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE char * FLMAPI f_strcpy(
		char *			pszDest,
		const char *	pszSrc)
	{
		return( strcpy( pszDest, pszSrc));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE FLMINT FLMAPI f_strlen(
		const char *	pszStr)
	{
		return( strlen( pszStr));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE FLMINT FLMAPI f_strcmp(
		const char *		pvStr1,
		const char *		pvStr2)
	{
		return( strcmp( pvStr1, pvStr2));
	}
		
	/****************************************************************************
	Desc:
	****************************************************************************/
	#ifdef FLM_WIN
	FINLINE FLMINT FLMAPI f_stricmp(
		const char *		pvStr1,
		const char *		pvStr2)
	{
		return( _stricmp( pvStr1, pvStr2));
	}
	#endif
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	FINLINE FLMINT FLMAPI f_strncmp(
		const char *		pvStr1,
		const char *		pvStr2,
		FLMSIZET				uiLength)
	{
		return( strncmp( pvStr1, pvStr2, uiLength));
	}
		
	/****************************************************************************
	Desc:
	****************************************************************************/
	#ifdef FLM_WIN
	FINLINE FLMINT FLMAPI f_strnicmp(
		const char *		pvStr1,
		const char *		pvStr2,
		FLMSIZET				uiLength)
	{
		return( _strnicmp( pvStr1, pvStr2, uiLength));#else
	}
	#endif

#if 0	
		#define f_strcat( dest, src) \
			strcat( (char*)(dest), (char*)(src))

		#define f_strchr( str, value) \
			strchr( (char*)str, (int)value)

		#define f_strncpy( dest, src, length) \
			strncpy( (char*)(dest), (char*)(src), (size_t)(length))

		#define f_strrchr( str, value ) \
			strrchr( (char*)(str), (int)value)

		#define f_strstr( str1, str2) \
			(char *)strstr( (char*)(str1), (char*)(str2))

		#define f_strncat( str1, str2, n) \
			strncat( (char *)(str1), (char *)(str2), n)

		#define f_strupr( str) \
			_strupr( (char *)(str))
#endif

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

		virtual FINLINE FLMINT FLMAPI AddRef( void)
		{
			return( ++m_refCnt);
		}

		virtual FINLINE FLMINT FLMAPI Release( void)
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
	class F_Base
	{
	public:
	
		F_Base()
		{
		}
	
		virtual ~F_Base()
		{
		}
	
		void * operator new(
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	
		void * operator new[](
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
		
		void operator delete(
			void *			ptr);
	
		void operator delete(
			void *			ptr,
			const char *	file,
			int				line);
	
		void operator delete[](
			void *			ptr,
			const char *	file,
			int				line);
	};

	/****************************************************************************
	Desc: Asserts
	****************************************************************************/

	#ifdef FLM_DEBUG

		#if defined( FLM_WIN)
		
			#define flmAssert( exp) \
				(void)( (exp) || (DebugBreak(), 0))

		#elif defined( FLM_NLM)
		
			extern "C" void EnterDebugger(void);

			#define flmAssert( exp) \
				(void)( (exp) || ( EnterDebugger(), 0))

		#elif defined( FLM_UNIX)
		
			#define flmAssert( exp)	\
				(void)( (exp) || (assert(0), 0))

		#else
			#define flmAssert( exp)
		#endif

	#else
		#define flmAssert( exp)
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
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	FINLINE FLMINT32 FLMAPI f_atomicInc(
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
		#elif defined( FLM_UNIX)
			return( posix_atomic_add_32( piTarget, 1));
		#else
			#error Atomic operations aren't supported
		#endif
	}
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	FINLINE FLMINT32 FLMAPI f_atomicDec(
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
		#elif defined( FLM_UNIX)
			return( posix_atomic_add_32( piTarget, -1));
		#else
			#error Atomic operations aren't supported
		#endif
	}
	
	/**********************************************************************
	Desc:
	**********************************************************************/
	FINLINE FLMINT32 FLMAPI f_atomicExchange(
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
		#elif defined( FLM_UNIX)
			return( posix_atomic_xchg_32( piTarget, i32NewVal));
		#else
			#error Atomic operations aren't supported
		#endif
	}

	/****************************************************************************
	Desc: Mutex and semaphore routines
	****************************************************************************/
	#ifdef FLM_NLM
		FINLINE RCODE FLMAPI f_mutexCreate(
			F_MUTEX *	phMutex)
		{
			if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"NOVDB")) == F_MUTEX_NULL)
			{
				return RC_SET( NE_FLM_MEM);
			}

			return NE_FLM_OK;
		}
	
		FINLINE void FLMAPI f_mutexDestroy(
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
	
		FINLINE void FLMAPI f_mutexLock(
			F_MUTEX		hMutex)
		{
			(void)kMutexLock( (MUTEX)hMutex);
		}
	
		FINLINE void FLMAPI f_mutexUnlock(
			F_MUTEX		hMutex)
		{
			(void)kMutexUnlock( (MUTEX)hMutex);
		}
	
		FINLINE void FLMAPI f_assertMutexLocked(
			F_MUTEX)
		{
		}

		typedef SEMAPHORE				F_SEM;
		typedef SEMAPHORE *			F_SEM_p;
		#define F_SEM_NULL			0

		FINLINE RCODE FLMAPI f_semCreate(
			F_SEM *		phSem)
		{
			if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"NOVDB", 0)) == F_SEM_NULL)
			{
				return RC_SET( NE_FLM_MEM);
			}
	
			return NE_FLM_OK;
		}
	
		FINLINE void FLMAPI f_semDestroy(
			F_SEM *		phSem)
		{
			if (*phSem != F_SEM_NULL)
			{
				(void)kSemaphoreFree( (SEMAPHORE)(*phSem));
				*phSem = F_SEM_NULL;
			}
		}
	
		FINLINE RCODE FLMAPI f_semWait(
			F_SEM			hSem,
			FLMUINT		uiTimeout)
		{
			RCODE			rc = NE_FLM_OK;
	
			if( uiTimeout == F_SEM_WAITFOREVER)
			{
				if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
				{
					rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
				}
			}
			else
			{
				if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
				{
					rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
				}
			}
	
			return( rc);
		}
	
		FINLINE void FLMAPI f_semSignal(
			F_SEM			hSem)
		{
			(void)kSemaphoreSignal( (SEMAPHORE)hSem);
		}
	
	#elif defined( FLM_WIN)
	
		typedef struct
		{
			FLMATOMIC						locked;
	#ifdef FLM_DEBUG
			FLMUINT							uiThreadId;
			FLMATOMIC						lockedCount;
			FLMATOMIC						waitCount;
	#endif
		} F_INTERLOCK;

		RCODE FLMAPI f_mutexCreate(
			F_MUTEX *	phMutex);
	
		void FLMAPI f_mutexDestroy(
			F_MUTEX *	phMutex);
			
		FINLINE void FLMAPI f_mutexLock(
			F_MUTEX		hMutex)
		{
			while( f_atomicExchange( 
				&(((F_INTERLOCK *)hMutex)->locked), 1) != 0)
			{
		#ifdef FLM_DEBUG
				f_atomicInc( &(((F_INTERLOCK *)hMutex)->waitCount));
		#endif
				Sleep( 0);
			}
	
		#ifdef FLM_DEBUG
			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == 0);
			((F_INTERLOCK *)hMutex)->uiThreadId = _threadid;
			f_atomicInc( &(((F_INTERLOCK *)hMutex)->lockedCount));
		#endif
		}
		
		FINLINE void FLMAPI f_mutexUnlock(
			F_MUTEX		hMutex)
		{
			flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
		#ifdef FLM_DEBUG
			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
			((F_INTERLOCK *)hMutex)->uiThreadId = 0;
		#endif
			f_atomicExchange( &(((F_INTERLOCK *)hMutex)->locked), 0);
		}
	
		FINLINE void FLMAPI f_assertMutexLocked(
			F_MUTEX		hMutex)
		{
		#ifdef FLM_DEBUG
			flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
		#else
			F_UNREFERENCED_PARM( hMutex);
		#endif
		}
		
		FINLINE RCODE FLMAPI f_semCreate(
			F_SEM *		phSem)
		{
			if( (*phSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL,
				0, 10000, NULL )) == NULL)
			{
				return( RC_SET( NE_FLM_COULD_NOT_CREATE_SEMAPHORE));
			}
	
			return NE_FLM_OK;
		}
	
		FINLINE void FLMAPI f_semDestroy(
			F_SEM *		phSem)
		{
			if (*phSem != F_SEM_NULL)
			{
				CloseHandle( *phSem);
				*phSem = F_SEM_NULL;
			}
		}
	
		FINLINE RCODE FLMAPI f_semWait(
			F_SEM			hSem,
			FLMUINT		uiTimeout)
		{
			if( WaitForSingleObject( hSem, uiTimeout ) == WAIT_OBJECT_0)
			{
				return( NE_FLM_OK);
			}
			else
			{
				return( RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE));
			}
		}
	
		FINLINE void FLMAPI f_semSignal(
			F_SEM			hSem)
		{
			(void)ReleaseSemaphore( hSem, 1, NULL);
		}
	#elif defined( FLM_UNIX)
		RCODE FLMAPI f_mutexCreate(
			F_MUTEX *	phMutex);
	
		void f_mutexDestroy(
			F_MUTEX *	phMutex);
		
		FINLINE void FLMAPI f_mutexLock(
			F_MUTEX		hMutex)
		{
			(void)pthread_mutex_lock( (pthread_mutex_t *)hMutex);
		}
	
		FINLINE void FLMAPI f_mutexUnlock(
			F_MUTEX		hMutex)
		{
			(void)pthread_mutex_unlock( (pthread_mutex_t *)hMutex);
		}
	
		FINLINE void FLMAPI f_assertMutexLocked(
			F_MUTEX)
		{
		}

		void FLMAPI f_semDestroy(
			F_SEM *	phSem);
	
		RCODE FLMAPI f_semCreate(
			F_SEM *	phSem);
	
		RCODE FLMAPI f_semWait(
			F_SEM		hSem,
			FLMUINT	uiTimeout);
	
	#endif

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
	class F_IOBuffer : public IF_IOBuffer, public F_Base
	{
	#define MAX_BUFFER_BLOCKS	16
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
			flmAssert( uiBlockNumber < MAX_BUFFER_BLOCKS);
			m_UserData [uiBlockNumber] = pvData;
		}
	
		FINLINE void * FLMAPI getCompletionCallbackData(
			FLMUINT	uiBlockNumber)
		{
			flmAssert( uiBlockNumber < MAX_BUFFER_BLOCKS);
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
	
	#ifdef FLM_NLM
		void signalComplete(
			RCODE		rc);
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
		void *					m_UserData [MAX_BUFFER_BLOCKS];
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
	#ifdef FLM_NLM
		F_SEM						m_hSem;
	#endif
		F_IOBuffer *			m_pNext;
		F_IOBuffer *			m_pPrev;
		WRITE_COMPLETION_CB	m_fnCompletion;
		RCODE						m_completionRc;
		F_TMSTAMP				m_StartTime;
		FLMUINT64				m_ui64ElapMilli;
	
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
	class F_FileHdl : public IF_FileHdl, public F_Base
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
				flmAssert( pvBufferObj == NULL);
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
		
		FINLINE void FLMAPI setBlockSize(
			FLMUINT	uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
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
	
		FINLINE FLMUINT getSectorSize( void)
		{
			return( m_uiBytesPerSector);
		}
	
		FINLINE HANDLE getFileHandle( void)
		{
			return m_FileHandle;
		}
	
		RCODE openOrCreate(
			const char *	pszFileName,
			FLMUINT			uiAccess,
			FLMBOOL			bCreateFlag);
	
		RCODE allocAlignBuffer( void);
	
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
	#ifdef FLM_UNIX
	class F_FileHdl : public IF_FileHdl, public F_Base
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
			FLMUINT				uiMaxAutoExtendSize)
		{
			m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
		}
	
		RCODE FLMAPI lock( void);
	
		RCODE FLMAPI unlock( void);
	
		FINLINE void FLMAPI setBlockSize(
			FLMUINT				uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
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
		
		RCODE allocAlignBuffer( void);
		
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
		FLMUINT					m_uiMaxAutoExtendSize;
		FLMBOOL					m_bCanDoAsync;
		FLMBOOL					m_bDoDirectIO;
		FLMBYTE *				m_pucAlignedBuff;
		FLMUINT					m_uiAlignedBuffSize;
		
		friend class F_FileSystem;
		friend class F_MultiFileHdl;
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
	class F_DirHdl : public IF_DirHdl, public F_Base
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
	Desc: Logging
	****************************************************************************/

	void flmDbgLogInit( void);
	void flmDbgLogExit( void);
	void flmDbgLogFlush( void);

	/****************************************************************************
	Desc:	Timers
	****************************************************************************/
	#if defined( FLM_NLM)
	
		extern "C" void ConvertTicksToSeconds(
			LONG		ticks,
			LONG *	seconds,
			LONG *	tenthsOfSeconds);
	
		extern "C" void ConvertSecondsToTicks(
			LONG		seconds,
			LONG		tenthsOfSeconds,
			LONG *	ticks);
	
		#define FLM_GET_TIMER()	\
			(FLMUINT)GetCurrentTime()
	
		#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU) \
			ConvertSecondsToTicks( (LONG)(uiSeconds), 0, (LONG *)(&(uiTU)))
	
		#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds) \
		{ \
			LONG	udDummy; \
			ConvertTicksToSeconds( (LONG)(uiTU), (LONG *)(&(uiSeconds)), &udDummy); \
		}
	
		#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli) \
		{ \
			LONG	udTenths; \
			LONG	udSeconds; \
			ConvertTicksToSeconds( (LONG)(uiTU), (LONG *)(&(udSeconds)), &udTenths); \
			uiMilli = (FLMUINT)(udSeconds) * 1000 + (FLMUINT)udTenths * 100; \
		}
		
		#define FLM_MILLI_TO_TIMER_UNITS( uiMilliSeconds, uiTU) \
		{ \
			LONG udTenths, udSeconds; \
			udSeconds = ((LONG) uiMilliSeconds) / 1000; \
			udTenths = (((LONG) uiMilliSeconds) % 1000) / 100; \
			ConvertSecondsToTicks( udSeconds, udTenths, (LONG *)(&(uiTU))); \
		}
	
	#elif defined( FLM_UNIX)
	
		// gettimeofday() is actually 4 times faster than time() on
		// Solaris. gethrtime() is even faster. On Linux time() is the
		// fastest; gettimeofday() is 50% slower. clock() is the
		// slowest on both Solaris and Linux. We use a new function for
		// millisec resolution. The implementation is OS dependent.
	
		#define FLM_GET_TIMER() \
			(FLMUINT) f_timeGetMilliTime()
			
		#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU) \
			 ((uiTU) = ((uiSeconds) * 1000))
			 
		#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds) \
			 ((uiSeconds) = ((uiTU) / 1000))
			 
		#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli) \
			 ((uiMilli) = (uiTU))
			 
		#define FLM_MILLI_TO_TIMER_UNITS( uiMilli, uiTU) \
			 ((uiTU) = (uiMilli))

	#elif defined( FLM_WIN)
	
		#define FLM_GET_TIMER() \
			(FLMUINT)GetTickCount()
	
		#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU) \
			((uiTU) = (uiSeconds) * 1000)
	
		#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds) \
			((uiSeconds) = (uiTU) / 1000)
	
		#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli) \
			(uiMilli = (uiTU))
	
		#define FLM_MILLI_TO_TIMER_UNITS( uiMilliSeconds, uiTU) \
			(uiTU = (uiMilliSeconds))
	
	#endif
	
	// This macro for calculating elapsed time accounts for the
	// possibility of the time wrapping - which it will for some
	// of our counters (FLM_WIN is milliseconds and wraps in 49.7 days).
	
	#define FLM_ELAPSED_TIME(uiLaterTime,uiEarlierTime) \
		(FLMUINT)(((uiLaterTime) >= (uiEarlierTime)) \
					 ? (FLMUINT)((uiLaterTime) - (uiEarlierTime)) \
					 : (FLMUINT)((0xFFFFFFFF - (uiEarlierTime)) + (uiLaterTime)))
				 
	/****************************************************************************
	Desc:	Misc.
	****************************************************************************/
	FLMUINT flmGetFSBlockSize(
		FLMBYTE *			pszFileName);
		
	#if defined( FLM_LINUX)
		void flmGetLinuxKernelVersion(
			FLMUINT *		puiMajor,
			FLMUINT *		puiMinor,
			FLMUINT *		puiRevision);
			
		FLMUINT flmGetLinuxMaxFileSize( void);
		
		void flmGetLinuxMemInfo(
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
		
#endif	// FTKSYS_H
