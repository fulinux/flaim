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
	
	#define F_SEM_WAITFOREVER			0xFFFFFFFF
	
	#ifdef FLM_NLM
		#include "ftknlm.h"
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
		#include <aio.h>
		
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

		#ifdef FLM_OSX
			#include <libkern/OSAtomic.h>
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
	
		#ifndef INVALID_SOCKET
			#define INVALID_SOCKET 		(-1)
		#endif
	
		#ifndef INADDR_NONE
			#define INADDR_NONE			(-1)
		#endif
	
		#ifndef SOCKET
			#define SOCKET					int
		#endif

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
	Desc: ASCII Constants
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
	
	/****************************************************************************
	Desc: Native constants
	****************************************************************************/
	
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
	
	#define f_toascii(native) \
		(native)
	
	#define f_tonative(ascii) \
		(ascii)
	
	#define f_toupper(native) \
		(((native) >= 'a' && (native) <= 'z') \
		? (native) - 'a' + 'A' : (native))
	
	#define f_tolower(native) \
		(((native) >= 'A' && (native) <= 'Z') \
		? (native) - 'A' + 'a' : (native))
	
	#define f_islower(native) \
		((native) >= 'a' && (native) <= 'z')
	
	/****************************************************************************
	Desc: Unicode constants
	****************************************************************************/
	
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
	Desc: Byte order macros
	****************************************************************************/
	
	FINLINE FLMUINT16 flmBigEndianToUINT16(
		FLMBYTE *		pucBuf)
	{
		FLMUINT16		ui16Val = 0;
		
		ui16Val |= ((FLMUINT16)pucBuf[ 0]) << 8;
		ui16Val |= ((FLMUINT16)pucBuf[ 1]);
		
		return( ui16Val);
	}
	
	FINLINE FLMUINT32 flmBigEndianToUINT32( 
		FLMBYTE *		pucBuf)
	{
		FLMUINT32		ui32Val = 0;

		ui32Val |= ((FLMUINT32)pucBuf[ 0]) << 24;
		ui32Val |= ((FLMUINT32)pucBuf[ 1]) << 16;
		ui32Val |= ((FLMUINT32)pucBuf[ 2]) << 8;
		ui32Val |= ((FLMUINT32)pucBuf[ 3]);
		
		return( ui32Val);
	}
	
	FINLINE FLMUINT64 flmBigEndianToUINT64( 
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
	
	FINLINE FLMINT16 flmBigEndianToINT16(
		FLMBYTE *		pucBuf)
	{
		FLMINT16		i16Val = 0;
		
		i16Val |= ((FLMINT16)pucBuf[ 0]) << 8;
		i16Val |= ((FLMINT16)pucBuf[ 1]);
		
		return( i16Val);
	}
	
	FINLINE FLMINT32 flmBigEndianToINT32( 
		FLMBYTE *		pucBuf)
	{
		FLMINT32			i32Val = 0;

		i32Val |= ((FLMINT32)pucBuf[ 0]) << 24;
		i32Val |= ((FLMINT32)pucBuf[ 1]) << 16;
		i32Val |= ((FLMINT32)pucBuf[ 2]) << 8;
		i32Val |= ((FLMINT32)pucBuf[ 3]);
		
		return( i32Val);
	}
	
	FINLINE FLMINT64 flmBigEndianToINT64( 
		FLMBYTE *		pucBuf)
	{
		FLMINT64			i64Val = 0;
	
		i64Val |= ((FLMINT64)pucBuf[ 0]) << 56;
		i64Val |= ((FLMINT64)pucBuf[ 1]) << 48;
		i64Val |= ((FLMINT64)pucBuf[ 2]) << 40;
		i64Val |= ((FLMINT64)pucBuf[ 3]) << 32;
		i64Val |= ((FLMINT64)pucBuf[ 4]) << 24;
		i64Val |= ((FLMINT64)pucBuf[ 5]) << 16;
		i64Val |= ((FLMINT64)pucBuf[ 6]) << 8;
		i64Val |= ((FLMINT64)pucBuf[ 7]);
	
		return( i64Val);
	}
	
	FINLINE void flmUINT32ToBigEndian( 
		FLMUINT32		ui32Num,
		FLMBYTE *		pucBuf)
	{
		pucBuf[ 0] = (FLMBYTE) (ui32Num >> 24);
		pucBuf[ 1] = (FLMBYTE) (ui32Num >> 16);
		pucBuf[ 2] = (FLMBYTE) (ui32Num >>  8);
		pucBuf[ 3] = (FLMBYTE) (ui32Num);
	}
	
	FINLINE void flmINT32ToBigEndian( 
		FLMINT32			i32Num,
		FLMBYTE *		pucBuf)
	{
		pucBuf[ 0] = (FLMBYTE) (i32Num >> 24);
		pucBuf[ 1] = (FLMBYTE) (i32Num >> 16);
		pucBuf[ 2] = (FLMBYTE) (i32Num >>  8);
		pucBuf[ 3] = (FLMBYTE) (i32Num);
	}
	
	FINLINE void flmINT64ToBigEndian( 
		FLMINT64			i64Num,
		FLMBYTE *		pucBuf)
	{
		pucBuf[ 0] = (FLMBYTE) (i64Num >> 56);
		pucBuf[ 1] = (FLMBYTE) (i64Num >> 48);
		pucBuf[ 2] = (FLMBYTE) (i64Num >> 40);
		pucBuf[ 3] = (FLMBYTE) (i64Num >> 32);
		pucBuf[ 4] = (FLMBYTE) (i64Num >> 24);
		pucBuf[ 5] = (FLMBYTE) (i64Num >> 16);
		pucBuf[ 6] = (FLMBYTE) (i64Num >>  8);
		pucBuf[ 7] = (FLMBYTE) (i64Num);
	}
	
	FINLINE void flmUINT64ToBigEndian( 
		FLMUINT64		ui64Num,
		FLMBYTE *		pucBuf)
	{
		pucBuf[ 0] = (FLMBYTE) (ui64Num >> 56);
		pucBuf[ 1] = (FLMBYTE) (ui64Num >> 48);
		pucBuf[ 2] = (FLMBYTE) (ui64Num >> 40);
		pucBuf[ 3] = (FLMBYTE) (ui64Num >> 32);
		pucBuf[ 4] = (FLMBYTE) (ui64Num >> 24);
		pucBuf[ 5] = (FLMBYTE) (ui64Num >> 16);
		pucBuf[ 6] = (FLMBYTE) (ui64Num >>  8);
		pucBuf[ 7] = (FLMBYTE) (ui64Num);
	}
	
	FINLINE void flmINT16ToBigEndian( 
		FLMINT16			i16Num,
		FLMBYTE *		pucBuf)
	{
		pucBuf[ 0] = (FLMBYTE) (i16Num >> 8);
		pucBuf[ 1] = (FLMBYTE) (i16Num);
	}
	
	FINLINE void flmUINT16ToBigEndian( 
		FLMUINT16		ui16Num,
		FLMBYTE *		pucBuf)
	{
		pucBuf[ 0] = (FLMBYTE) (ui16Num >> 8);
		pucBuf[ 1] = (FLMBYTE) (ui16Num);
	}
	
	#ifndef FLM_BIG_ENDIAN
	
		#if defined( FLM_SPARC) || defined( FLM_POWER_PC)
			#error Wrong endian order selected
		#endif
	
		#define LO(wrd) \
			(*(FLMUINT8 *)&wrd)
			
		#define HI(wrd) \
			(*((FLMUINT8 *)&wrd + 1))
	
		#if( defined( FLM_UNIX) && defined( FLM_STRICT_ALIGNMENT))
	
			#define FB2UW( bp) \
				((FLMUINT16)((((FLMUINT16)(((FLMUINT8 *)(bp))[1]))<<8) | \
					(((FLMUINT16)(((FLMUINT8 *)(bp))[0])))))
	
			#define FB2UD( bp) \
				((FLMUINT32)(	(((FLMUINT32)(((FLMUINT8 *)(bp))[3]))<<24) | \
					(((FLMUINT32)(((FLMUINT8 *)(bp))[2]))<<16) | \
					(((FLMUINT32)(((FLMUINT8 *)(bp))[1]))<< 8) | \
					(((FLMUINT32)(((FLMUINT8 *)(bp))[0])))))
	
			#define UW2FBA( uw, bp) \
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)(uw)), \
					((FLMUINT8 *)(bp))[1] = ((FLMUINT8)((((uw) & 0xff00)>>8))))
	
			#define UD2FBA( udw, bp) \
				(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((udw) & 0xff)), \
					 ((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((udw) & 0xff00)>>8)), \
					 ((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((udw) & 0xff0000)>>16)), \
					 ((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((udw) & 0xff000000)>>24)))
	
		#else
	
			#define FB2UW( fbp) \
				(*((FLMUINT16 *)(fbp)))
				
			#define FB2UD( fbp) \
				(*((FLMUINT32 *)(fbp)))
				
			#define UW2FBA( uw, fbp) \
				(*((FLMUINT16 *)(fbp)) = ((FLMUINT16) (uw)))
				
			#define UD2FBA( uw, fbp) \
				(*((FLMUINT32 *)(fbp)) = ((FLMUINT32) (uw)))
	
		#endif
		
	#else
	
		#if defined( __i386__)
			#error Wrong endian order selected
		#endif
	
		#define LO(wrd) \
			(*((FLMUINT8 *)&wrd + 1))
		
		#define HI(wrd) \
			(*(FLMUINT8  *)&wrd)
	
		#define FB2UW( bp) \
			((FLMUINT16)((((FLMUINT16)(((FLMUINT8 *)(bp))[1])) << 8) | \
					(((FLMUINT16)(((FLMUINT8 *)(bp))[0]))   ) ))
	
		#define FB2UD( bp) \
			((FLMUINT32)((((FLMUINT32)(((FLMUINT8 *)(bp))[3])) << 24) | \
					(((FLMUINT32)(((FLMUINT8 *)(bp))[2])) << 16) | \
					(((FLMUINT32)(((FLMUINT8 *)(bp))[1])) << 8) | \
					(((FLMUINT32)(((FLMUINT8 *)(bp))[0]))    ) ))
	
		#define UW2FBA( uw, bp)	\
			(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)(uw)), \
					((FLMUINT8 *)(bp))[1] = ((FLMUINT8)((((uw) & 0xff00)>>8))))
	
		#define UD2FBA( udw, bp) \
			(((FLMUINT8 *)(bp))[0] = ((FLMUINT8)((udw) & 0xff)), \
					((FLMUINT8 *)(bp))[1] = ((FLMUINT8)(((udw) & 0xff00)>>8)), \
					((FLMUINT8 *)(bp))[2] = ((FLMUINT8)(((udw) & 0xff0000)>>16)), \
					((FLMUINT8 *)(bp))[3] = ((FLMUINT8)(((udw) & 0xff000000)>>24)))
	#endif
	
	/****************************************************************************
	Desc: File Path Functions & Macros
	****************************************************************************/
	
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
	Desc: CPU Release Functions										
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
	
	/****************************************************************************
	Desc: Mutexes
	****************************************************************************/
	
	#if defined( FLM_WIN)
	
		RCODE f_mutexCreate(
			F_MUTEX *	phMutex);
	
		void f_mutexDestroy(
			F_MUTEX *	phMutex);
			
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
	
		RCODE f_mutexCreate(
			F_MUTEX *	phMutex);
	
		void f_mutexDestroy(
			F_MUTEX *	phMutex);
			
		void f_mutexLock(
			F_MUTEX		hMutex);
		
		void f_mutexUnlock(
			F_MUTEX		hMutex);
	
	#endif
	
	
	/****************************************************************************
	Desc: Semaphores
	****************************************************************************/
	
	#if defined( FLM_WIN)
	
		typedef HANDLE					F_SEM;
		typedef HANDLE *				F_SEM_p;
		#define F_SEM_NULL			NULL
	
	#elif defined( FLM_UNIX)
	
		#if defined( FLM_AIX) || defined( FLM_OSX)
		
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
			
		#endif
		
		typedef F_SEM *				F_SEM_p;
		#define F_SEM_NULL			NULL
	
	#elif !defined( FLM_NLM)
		#error Unsupported platform
	#endif
	
	#if defined( FLM_WIN)
	
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
		#elif defined( FLM_GNUC)
		{
			#if defined( __i386__) || defined( __x86_64__)
			{
				FLMINT32 			i32Tmp;
				
				__asm__ __volatile__ (
								"lock; xaddl %0, %1"
								: "=r" (i32Tmp), "=m" (*piTarget)
								: "0" (1), "m" (*piTarget));
			
				return( i32Tmp + 1);
			}
			#elif defined( __ppc__) || defined ( __powerpc__)
			{
				FLMINT32				i32Result = 0;
				FLMINT32				i32Tmp;
			
				__asm__ __volatile__ (
								"sync;"
								"1:		lwarx  %0, 0, %2;"
								"			addi %1, %0, 1;"
								"			stwcx. %1, 0, %2;"
								"			bne-   1b;"
								"sync"
								: "=&b" (i32Result), "=&b" (i32Tmp) 
								: "r" (piTarget) 
								: "cc", "memory");
		
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
				
				__asm__ __volatile__ (
								"\tLA\t2,%1\n"
								"0:\tL\t%0,%1\n"
								"\tLR\t1,%0\n"
								"\tAHI\t1,1\n"
								"\tCS\t%0,1,0(2)\n"
								"\tJNZ\t0b\n"
								"\tLR\t%0,1"
								: "=r" (i32Tmp), "+m" (*piTarget)
								: 
								: "1", "2", "cc");
			
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
			
				__asm__ __volatile__ (
								"sync;"
								"1:		lwarx  %0, 0, %2;"
								"			addi %1, %0, -1;"
								"			stwcx. %1, 0, %2;"
								"			bne-   1b;"
								"sync"
								: "=&b" (i32Result), "=&b" (i32Tmp) 
								: "r" (piTarget) 
								: "cc", "memory");
								
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
		
				__asm__ __volatile__ (
								"\tLA\t2,%1\n"
								"0:\tL\t%0,%1\n"
								"\tLR\t1,%0\n"
								"\tAHI\t1,-1\n"
								"\tCS\t%0,1,0(2)\n"
								"\tJNZ\t0b\n"
								"\tLR\t%0,1"
								: "=r" (i32Tmp), "+m" (*piTarget)
								: 
								: "1", "2", "cc");
			
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

				if( OSAtomicCompareAndSwap32( iOldVal, i32NewVal, (int32_t *)piTarget))
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
				FLMINT32 			i32OldVal;
				
				__asm__ __volatile__ ("1:; lock; cmpxchgl %2, %0; jne 1b"
								: "=m" (*piTarget), "=a" (i32OldVal)
								: "r" (i32NewVal), "m" (*piTarget),
								  "a" (*piTarget));
			
				return( i32OldVal);
			}
			#elif defined( __ppc__) || defined ( __powerpc__)
			{
				FLMINT32				i32OldVal = 0;
			
				__asm__ __volatile__ ( 
								"sync;"
								"0:		lwarx %0,0,%1;"
								"			xor. %0,%3,%0;"
								"			bne 1f;"
								"			stwcx. %2,0,%1;"
								"			bne- 0b;"
								"1:		sync"
								: "=&r" (i32OldVal) 
								: "r" (piTarget), "r" (i32NewVal), "r" (*piTarget) 
								: "cr0", "memory");
								
				return( i32OldVal);
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
				
				__asm__ __volatile__ (
								"\tLA\t1,%0\n"
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
	
	/***************************************************************************
	Desc:
	****************************************************************************/
	FINLINE FLMUINT64 flmRoundUp(
		FLMUINT64		ui64ValueToRound,
		FLMUINT64		ui64Boundary)
	{
		FLMUINT64	ui64RetVal;
		
		ui64RetVal = ((ui64ValueToRound / ui64Boundary) * ui64Boundary);	
		
		if( ui64RetVal < ui64ValueToRound)
		{
			ui64RetVal += ui64Boundary;
		}
		
		return( ui64RetVal);
	}
	
	/****************************************************************************
	Desc: Process ID Functions
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
	Desc: Module Load/Unload Functions
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
