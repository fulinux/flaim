//-------------------------------------------------------------------------
// Desc:	Miscellaneous toolkit functions.
// Tabs:	3
//
//		Copyright (c) 2000-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftkmisc.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FLMUINT					gv_uiSerialInitCount = 0;
f_randomGenerator		gv_uiSerialRandom;
F_MUTEX					gv_hSerialMutex = F_MUTEX_NULL;

#if defined( FLM_NLM)

	#pragma pack(push,1) 

	extern "C"
	{
		LONG												gv_lMyModuleHandle = 0;
		LONG												gv_lFlmTimerTag = 0;
		LONG												gv_lAllocRTag = 0;
		static f_randomGenerator					gv_flmRandGenerator;
		static SEMAPHORE								gv_lFlmRandSemaphore = F_SEM_NULL;
		static LONG										gv_lFlmStartTicks = 0;
		static FLMUINT32								gv_ui32NetWareStartupCount = 0;
	}

	#pragma pack(pop)

	void f_sleep( 
		FLMUINT	uiMilliseconds)
	{
		if( ! uiMilliseconds )
		{
			kYieldThread();
		}
		else
		{
			kDelayThread( uiMilliseconds);
		}
	}

#endif


#if defined( FLM_UNIX)

	#ifdef FLM_AIX
		#ifndef nsleep
			extern "C"
			{
				extern int nsleep( struct timestruc_t *, struct timestruc_t *);
			}
		#endif
	#endif

/****************************************************************************
Desc: This routine causes the calling process to delay the given number
		of milliseconds.  Due to the nature of the call, the actual sleep
		time is almost guaranteed to be different from requested sleep time.
In:   milliseconds - the number of milliseconds to delay
****************************************************************************/
void f_sleep(
	FLMUINT		uiMilliseconds)
{
#ifdef FLM_AIX
	struct timestruc_t timeout;
	struct timestruc_t remain;
#else
	struct timespec timeout;
#endif

	timeout.tv_sec = (uiMilliseconds / 1000);
	timeout.tv_nsec = (uiMilliseconds % 1000) * 1000000;

#ifdef FLM_AIX
	nsleep(&timeout, &remain);
#else
	nanosleep(&timeout, 0);
#endif
}
#endif

#ifdef FLM_UNIX
/***************************************************************************
Desc:   Map POSIX errno to Flaim IO errors.
***************************************************************************/
RCODE MapErrnoToFlaimErr(
	int	err,
	RCODE	defaultRc)
{
	/* Switch on passed in error code value */

	switch (err)
	{
		case 0:
			return( FERR_OK);

		case ENOENT:
			return( RC_SET( FERR_IO_PATH_NOT_FOUND));

		case EACCES:
		case EEXIST:
			return( RC_SET( FERR_IO_ACCESS_DENIED));

		case EINVAL:
			flmAssert( 0);
			return( RC_SET( FERR_INVALID_PARM));

		case EIO:
			return( RC_SET( FERR_IO_DISK_FULL));

		case ENOTDIR:
			return( RC_SET( FERR_IO_MODIFY_ERR));

#ifdef EBADFD
		case EBADFD:
			return( RC_SET( FERR_IO_BAD_FILE_HANDLE));
#endif

		case EOF:
			return( RC_SET( FERR_IO_END_OF_FILE));

		case EMFILE:
			return( RC_SET( FERR_IO_NO_MORE_FILES));

		default:
			return( RC_SET( defaultRc));
	}
}
#endif

/****************************************************************************
Desc:		This routine initializes the serial number generator.  If the O/S
			does not provide support for GUID generation or if the GUID
			routines fail for some reason, a pseudo-GUID will be generated.
Notes:	This routine should only be called once by the process.
****************************************************************************/
RCODE f_initSerialNumberGenerator( void)
{
	FLMUINT					uiTime;
	RCODE						rc = FERR_OK;

	if (++gv_uiSerialInitCount > 1)
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &gv_hSerialMutex)))
	{
		goto Exit;
	}

	f_timeGetSeconds( &uiTime );

#if defined( FLM_WIN)
	f_randomSetSeed( &gv_uiSerialRandom,
		(FLMUINT32)(uiTime ^ (FLMUINT)_getpid())); 
#elif defined( FLM_NLM)
	f_randomSetSeed( &gv_uiSerialRandom,
		(FLMUINT32)(uiTime ^ (FLMUINT)GetRunningProcess())); 
#elif defined( FLM_UNIX)
	f_randomSetSeed( &gv_uiSerialRandom,
		(FLMUINT32)(uiTime ^ (FLMUINT)getpid())); 
#else
	f_randomSetSeed( &gv_uiSerialRandom, (FLMUINT32)uiTime);
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:		This routine will use the operating system calls to generate a
			"globally unique" identifier.  Typically, this is based on the
			MAC address of an ethernet card installed in the machine.  If the
			machine does not have an ethernet card, or if the OS does not
			support generating GUIDs, this routine will generate a pseudo-GUID
			using a random number generator.  A serial number is 16-bytes.
****************************************************************************/
RCODE f_createSerialNumber(
	FLMBYTE *		pszSerialNum)
{
	RCODE						rc = FERR_OK;

#if defined( FLM_WIN)

	UUID			uuidVal;
	RPC_STATUS	err = UuidCreate( &uuidVal);

	if (err == RPC_S_OK || err == RPC_S_UUID_LOCAL_ONLY)
	{
		UD2FBA( (FLMUINT32)uuidVal.Data1, &pszSerialNum[ 0]);
		UW2FBA( (FLMUINT16)uuidVal.Data2, &pszSerialNum[ 4]);
		UW2FBA( (FLMUINT16)uuidVal.Data3, &pszSerialNum[ 6]);
		f_memcpy( &pszSerialNum[ 8], (FLMBYTE *)uuidVal.Data4, 8);
		goto Exit;
	}

#elif defined( FLM_NLM)

	NWGUID	guidVal;
	int		err = SGUIDCreate( &guidVal);

	if( !err || err == 1) // NOTE: 1 == SGUID_WARN_RANDOM_NODE
	{
		UD2FBA( guidVal.time_low, &pszSerialNum[ 0]);
		UW2FBA( guidVal.time_mid, &pszSerialNum[ 4]);
		UW2FBA( guidVal.time_hi_and_version, &pszSerialNum[ 6]);
		pszSerialNum[ 8] = guidVal.clk_seq_hi_res;
		pszSerialNum[ 9] = guidVal.clk_seq_low;
		f_memcpy( &pszSerialNum[ 10], (FLMBYTE *)guidVal.node, 6);
		goto Exit;
	}

#endif

	/*
	Generate a pseudo GUID value
	*/

	flmAssert( gv_hSerialMutex != F_MUTEX_NULL);

	f_mutexLock( gv_hSerialMutex);

	UD2FBA( (FLMUINT32)f_randomLong( 
		&gv_uiSerialRandom), &pszSerialNum[ 0]);
	UD2FBA( (FLMUINT32)f_randomLong( 
		&gv_uiSerialRandom), &pszSerialNum[ 4]);
	UD2FBA( (FLMUINT32)f_randomLong( 
		&gv_uiSerialRandom), &pszSerialNum[ 8]);
	UD2FBA( (FLMUINT32)f_randomLong( 
		&gv_uiSerialRandom), &pszSerialNum[ 12]);
	
	f_mutexUnlock( gv_hSerialMutex);

#if defined( FLM_WIN) || defined( FLM_NLM)
Exit:
#endif

	return( rc);
}

/****************************************************************************
Notes:	This routine should only be called once by the process.
****************************************************************************/
void f_freeSerialNumberGenerator( void)
{
	if( (--gv_uiSerialInitCount) > 0)
	{
		return;
	}

	if( gv_hSerialMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hSerialMutex);
	}
}


/****************************************************************************
Desc: Generates a table of remainders for each 8-bit byte.  The resulting
		table is used by flmUpdateCRC to calculate a CRC value.  The table
		must be freed via a call to f_freeCRCTable.
*****************************************************************************/
RCODE f_initCRCTable(
	FLMUINT32 **	ppui32CRCTbl)
{
	FLMUINT32 *		pTable;
	FLMUINT32		ui32Val;
	FLMUINT32		ui32Loop;
	FLMUINT32		ui32SubLoop;
	RCODE				rc = FERR_OK;

	// Use the standard degree-32 polynomial used by 
	// Ethernet, PKZIP, etc. for computing the CRC of
	// a data stream.  This is the little-endian
	// representation of the polynomial.  The big-endian
	// representation is 0x04C11DB7.

#define CRC_POLYNOMIAL		((FLMUINT32)0xEDB88320)

	*ppui32CRCTbl = NULL;

	if( RC_BAD( rc = f_alloc( 256 * sizeof( FLMUINT32), &pTable)))
	{
		goto Exit;
	}

	for( ui32Loop = 0; ui32Loop < 256; ui32Loop++)
	{
		ui32Val = ui32Loop;
		for( ui32SubLoop = 0; ui32SubLoop < 8; ui32SubLoop++)
		{
			if( ui32Val & 0x00000001)
			{
				ui32Val = CRC_POLYNOMIAL ^ (ui32Val >> 1);
			}
			else
			{
				ui32Val >>= 1;
			}
		}

		pTable[ ui32Loop] = ui32Val;
	}

	*ppui32CRCTbl = pTable;
	pTable = NULL;

Exit:

	if( pTable)
	{
		f_free( &pTable);
	}

	return( rc);
}

/****************************************************************************
Desc: Computes the CRC of the passed-in data buffer.  Multiple calls can
		be made to this routine to build a CRC over multiple data buffers.
		On the first call, *pui32CRC must be initialized to something 
		(0, etc.).  For generating CRCs that are compatible with PKZIP, 
		*pui32CRC should be initialized to 0xFFFFFFFF and the ones complement
		of the resulting CRC should be computed.
*****************************************************************************/
void f_updateCRC(
	FLMUINT32 *		pui32CRCTbl,
	FLMBYTE *		pucBlk,
	FLMUINT			uiBlkSize,
	FLMUINT32 *		pui32CRC)
{
	FLMUINT32		ui32CRC = *pui32CRC;
	FLMUINT			uiLoop;

	for( uiLoop = 0; uiLoop < uiBlkSize; uiLoop++)
	{
		ui32CRC = (ui32CRC >> 8) ^ pui32CRCTbl[ 
			((FLMBYTE)(ui32CRC & 0x000000FF)) ^ pucBlk[ uiLoop]];
	}

	*pui32CRC = ui32CRC;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT f_breakpoint(
	FLMUINT		uiBreakFlag)
{
	if( uiBreakFlag)
	{
#ifdef FLM_NLM
		EnterDebugger();
#else
		flmAssert( 0);
#endif
	}

	return( 0);
}

/****************************************************************************
Desc: 	Function that must be called within a NLM's startup routine.
****************************************************************************/
#ifdef FLM_NLM
RCODE f_netwareStartup( void)
{
	RCODE		rc = FERR_OK;

	if( ftkAtomicIncrement( &gv_i32NetWareStartupCount) != 1)
	{
		goto Exit;
	}

	gv_lMyModuleHandle = CFindLoadModuleHandle( (void *)f_netwareShutdown);

	// Allocate the needed resource tags

	if( (gv_lAllocRTag = AllocateResourceTag(
						gv_lMyModuleHandle,
						(BYTE *)"NOVDB Memory", AllocSignature)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (gv_lFlmTimerTag = AllocateResourceTag(
							gv_lMyModuleHandle,
							(BYTE *)"NOVDB Timer", TimerSignature)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	gv_lFlmStartTicks = GetCurrentTime();

	// Random Generator initialization

	if( (gv_lFlmRandSemaphore = kSemaphoreAlloc( 
		(BYTE *)"NOVDB", 1)) == F_SEM_NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	f_randomSetSeed( &gv_flmRandGenerator, 1);

Exit:

	if( RC_BAD( rc))
	{
		f_netwareShutdown();
	}

	return( rc);
}
#endif

/****************************************************************************
Desc: 	Closes (Frees) any resources used by FLAIM's clib patches layer.
****************************************************************************/
#ifdef FLM_NLM
void f_netwareShutdown( void)
{
	// Call exit function.

	if( ftkAtomicDecrement( &gv_ui32NetWareStartupCount) != 0)
	{
		goto Exit;
	}

	if( gv_lAllocRTag)
	{
		ReturnResourceTag( gv_lAllocRTag, 1);
		gv_lAllocRTag = 0;
	}

	if( gv_lFlmTimerTag)
	{
		ReturnResourceTag( gv_lFlmTimerTag, 1);
		gv_lFlmTimerTag = 0;
	}

	if( gv_lFlmRandSemaphore)
	{
		kSemaphoreFree(  gv_lFlmRandSemaphore);
		gv_lFlmRandSemaphore = 0;
	}

	gv_lFlmStartTicks = 0;
	gv_lMyModuleHandle = 0;

Exit:

	return;
}
#endif

/****************************************************************************
Desc: 	
****************************************************************************/
#ifdef FLM_NLM
FLMUINT f_getNLMHandle( void)
{
	return( (FLMUINT)gv_lMyModuleHandle);
}	
#endif

/****************************************************************************
Desc:		This routine is required to work around known bugs or inefficiencies 
			in various incarnations of clib.
****************************************************************************/
#undef f_memset
void * f_memset(
	void *			pvMem,
	FLMBYTE			ucByte,
	FLMUINT			uiSize)
{
#ifndef FLM_NLM
	return( memset( pvMem, ucByte, uiSize));
#else
	char *			cp = (char *)pvMem;
	unsigned			dwordLength;
	unsigned long	dwordVal;

	dwordVal = ((unsigned long)ucByte << 24) |
		((unsigned long)ucByte << 16) |
		((unsigned long)ucByte << 8) |
		(unsigned long)ucByte;

	while( uiSize && ((long)cp & 3L))
	{
		*cp++ = (char)ucByte;
		uiSize--;
	}

	dwordLength = uiSize >> 2;
	if(  dwordLength != 0)
	{
		CSetD( dwordVal, (void *)cp, dwordLength);
		cp += (dwordLength << 2);
		uiSize -= (dwordLength << 2);
	}

	while( uiSize)
	{
		*cp++ = (char)ucByte;
		uiSize--;
	}

	return( pvMem);
#endif
}

/****************************************************************************
Desc:		This routine is required to work around known bugs or inefficiencies 
			in various incarnations of clib.
****************************************************************************/
#undef f_memmove
void * f_memmove(
	void *			pvDest,
	const void *	pvSrc,
	FLMUINT			uiSize)
{
#ifndef FLM_NLM
	return( memmove( pvDest, pvSrc, uiSize));
#else
#define CMOVB_THRESHOLD		16
	char			*s = (char *)pvSrc;
	char			*d = (char *)pvDest;
	unsigned		uDiff;

	if( (char *)(s + uiSize) < d || (char *)(d + uiSize) < s)
	{
		// The source and destination do not overlap.

		CMoveFast( (void *)s, d, (LONG)uiSize);
	}
	else if( s < d)
	{
		// Source preceeds the destination, with overlap.

		uDiff = (unsigned)(d - s);
		d += uiSize;
		s += uiSize;
		if( uDiff >= CMOVB_THRESHOLD)
		{
			for( ;;)
			{
				if( uiSize < uDiff)
				{
					break;
				}

				// Copy the tail

				s -= uDiff;
				d -= uDiff;
				uiSize -= uDiff;
				CMoveFast( (void *)s, d, (LONG)uDiff);
			}
		}

		// Copy remaining bytes.

		while( uiSize--)
		{
			*--d = *--s;
		}
	}
	else if( s > d)
	{
		// Source follows the destination, with overlap.

		uDiff = (unsigned)(s - d);
		if( uDiff >= CMOVB_THRESHOLD)
		{
			for( ;;)
			{
				if( uiSize < uDiff)
				{
					break;
				}

				// Copy the head

				CMoveFast( (void *)s, d, (LONG)uDiff);
				uiSize -= uDiff;
				d += uDiff;
				s += uDiff;
			}
		}

		// Copy the remaining bytes

		while( uiSize--)
		{
			*d++ = *s++;
		}
	}

	// Else, the regions overlap completely (s == d).  Do nothing.

	return( pvDest);
#endif
}

/****************************************************************************
Desc:		Performs a comparison of m1 to m2, for a maximum
			length of size bytes.
****************************************************************************/
#undef f_memcmp
FLMINT f_memcmp(
	const void *	pvMem1,
	const void *	pvMem2,
	FLMUINT			uiSize)
{
	unsigned char *	s1;
	unsigned char *	s2;

	for (s1 = (unsigned char *)pvMem1, s2 = (unsigned char *)pvMem2; 
		uiSize > 0; uiSize--, s1++, s2++)
	{
		if (*s1 == *s2)
		{
			continue;
		}
		else if( *s1 > *s2)
		{
			return( 1);
		}
		else
		{
			return( -1);
		}
	}

	return( 0);
}

/****************************************************************************
Desc:		This routine is required to work around known bugs or inefficiencies 
			in various incarnations of clib.
****************************************************************************/
#undef f_stricmp
FLMINT f_stricmp(
	const char *	pszStr1,
	const char *	pszStr2)
{
	while( f_toupper( *pszStr1) == f_toupper( *pszStr2) && *pszStr1)
	{
		pszStr1++;
		pszStr2++;
	}
	return( (FLMINT)( f_toupper( *pszStr1) - f_toupper( *pszStr2)));
}

/****************************************************************************
Desc:		Performs a signed comparison of s1 to s2, for a maximum
			length of n bytes, starting with the first character in 
			each string and continuing with subsequent characters until 
			the corresponding characters differ, or until n characters 
			have been examined.
****************************************************************************/
#undef f_strnicmp
FLMINT f_strnicmp(
	const char *	pszStr1,
	const char *	pszStr2,
	FLMINT			iLen)
{
	if( !pszStr1 || !pszStr2)
	{
		return( (pszStr1 == pszStr2) 
						? 0 
						: (pszStr1 ? 1 : -1));
	}

	while( iLen-- && *pszStr1 && *pszStr2 && 
		(f_toupper( *pszStr1) == f_toupper( *pszStr2)))
	{
		pszStr1++;
		pszStr2++;
	}

	return(	(iLen == -1)
					?	0
					:	(f_toupper( *pszStr1) - f_toupper( *pszStr2)));

}

/****************************************************************************
Desc: 	
****************************************************************************/
#undef f_strupr
char * f_strupr(
	char *		pszStr)
{
	while( *pszStr)
	{
		*pszStr = f_toupper( *pszStr);
		pszStr++;
	}

	return( pszStr);
}

/****************************************************************************
Desc: 	
****************************************************************************/
#undef f_strstr
char * f_strstr(
	const char *	pszStr1,
	const char *	pszStr2)
{
	FLMUINT 			i;
	FLMUINT			j;
	FLMUINT			k;

	if ( !pszStr1 || !pszStr2)
	{
		return( NULL);
	}

	for( i = 0; pszStr1[i] != '\0'; i++)
	{
		for( j=i, k=0; pszStr2[k] != '\0' &&
			pszStr1[j] == pszStr2[k]; j++, k++)
		{
			;
		}

		if ( k > 0 && pszStr2[k] == '\0')
		{
			return( (char *)&pszStr1[i]);
		}
	}

	return( NULL);
}

/****************************************************************************
Desc: 	
****************************************************************************/
#undef f_strchr
char * f_strchr(
	const char *	pszStr,
	char				c)
{
	if( !pszStr)
	{
		return( NULL);
	}

	while (*pszStr && *pszStr != (FLMBYTE)c)
	{
		pszStr++;
	}

	return( (char *)((*pszStr == c) 
								? pszStr
								: NULL));
}
