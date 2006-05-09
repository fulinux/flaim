//------------------------------------------------------------------------------
// Desc:	This file contains misc toolkit functions
//
// Tabs:	3
//
//		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftkmisc.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

static FLMUINT						gv_uiStartupCount = 0;
static FLMUINT						gv_uiSerialInitCount = 0;
static F_MUTEX						gv_hSerialMutex = F_MUTEX_NULL;
static IF_RandomGenerator *	gv_pSerialRandom = NULL;
static FLMUINT32 *				gv_pui32CRCTbl = NULL;
static IF_ThreadMgr *			gv_pThreadMgr = NULL;
static IF_FileSystem *			gv_pFileSystem = NULL;
static FLMUINT						gv_uiMaxFileSize = FLM_MAXIMUM_FILE_SIZE;
static F_XML *						gv_pXml = NULL;

FSTATIC RCODE f_initSerialNumberGenerator( void);

FSTATIC void f_freeSerialNumberGenerator( void);

FSTATIC RCODE f_initCRCTable(
	FLMUINT32 **	ppui32CRCTbl);

#ifdef FLM_AIX
	#ifndef nsleep
		extern "C"
		{
			extern int nsleep( struct timestruc_t *, struct timestruc_t *);
		}
	#endif
#endif

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE gv_ucSENLengthArray[] =
{
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 0   - 15
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 16  - 31
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 32  - 47
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 48  - 63
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 64  - 79
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 80  - 95
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 96  - 111
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,		// 112 - 127
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 128 - 143
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 144 - 159
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 160 - 175
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,		// 176 - 191
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,		// 192 - 207
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,		// 208 - 223
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,		// 224 - 239
	5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9		// 240 - 255
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE ucSENPrefixArray[] =
{
	0,
	0,
	0x80,
	0xC0,
	0xE0,
	0xF0,
	0xF8,
	0xFC,
	0xFE,
	0xFF
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI ftkStartup( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( ++gv_uiStartupCount > 1)
	{
		goto Exit;
	}
	
	f_memoryInit();
	
	if( RC_BAD( rc = f_initCharMappingTables()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_allocFileSystem( &gv_pFileSystem)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_allocThreadMgr( &gv_pThreadMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_initSerialNumberGenerator()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = f_initCRCTable( &gv_pui32CRCTbl)))
	{
		goto Exit;
	}
	
	if( (gv_pXml = f_new F_XML) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
		
	if( RC_BAD( rc = gv_pXml->setup()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_checkErrorCodeTables()))
	{
		goto Exit;
	}
	
#ifdef FLM_DEBUG
	if( RC_BAD( rc = f_verifyMetaphoneRoutines()))
	{
		goto Exit;
	}
#endif
	
#if defined( FLM_LINUX)
	f_setupLinuxKernelVersion();
	gv_uiMaxFileSize = f_getLinuxMaxFileSize();
#elif defined( FLM_AIX)

	// Call setrlimit to increase the max allowed file size.
	// We don't have a good way to deal with any errors returned by
	// setrlimit(), so we just hope that there aren't any ...
	
	struct rlimit rlim;
	
	rlim.rlim_cur = RLIM_INFINITY;
	rlim.rlim_max = RLIM_INFINITY;
	
	setrlimit( RLIMIT_FSIZE, &rlim);
#endif
	

Exit:

	if( RC_BAD( rc))
	{
		ftkShutdown();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI ftkShutdown( void)
{
	if( !gv_uiStartupCount || --gv_uiStartupCount > 0)
	{
		return;
	}
	
	if( gv_pThreadMgr)
	{
		gv_pThreadMgr->Release();
		gv_pThreadMgr = NULL;
	}
	
	if( gv_pFileSystem)
	{
		gv_pFileSystem->Release();
		gv_pFileSystem = NULL;
	}
	
	if( gv_pui32CRCTbl)
	{
		f_free( &gv_pui32CRCTbl);
	}
	
	if( gv_pXml)
	{
		gv_pXml->Release();
	}
	
	f_freeSerialNumberGenerator();
	f_freeCharMappingTables();
	f_memoryCleanup();
}

/****************************************************************************
Desc: This routine causes the calling process to delay the given number
		of milliseconds.  Due to the nature of the call, the actual sleep
		time is almost guaranteed to be different from requested sleep time.
****************************************************************************/
#ifdef FLM_UNIX
void FLMAPI f_sleep(
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

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FLMAPI f_sleep(
	FLMUINT		uiMilliseconds)
{
	Sleep( (DWORD)uiMilliseconds);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_NLM
void FLMAPI f_sleep( 
	FLMUINT		uiMilliseconds)
{
	if( !uiMilliseconds )
	{
		pthread_yield();
	}
	else
	{
		delay( uiMilliseconds);
	}
}
#endif

/****************************************************************************
Desc:		This routine initializes the serial number generator.  If the O/S
			does not provide support for GUID generation or if the GUID
			routines fail for some reason, a pseudo-GUID will be generated.
Notes:	This routine should only be called once by the process.
****************************************************************************/
FSTATIC RCODE f_initSerialNumberGenerator( void)
{
	FLMUINT					uiTime;
	RCODE						rc = NE_FLM_OK;

	if (++gv_uiSerialInitCount > 1)
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &gv_hSerialMutex)))
	{
		goto Exit;
	}

	f_timeGetSeconds( &uiTime );

#if defined( FLM_UNIX) || defined( FLM_NLM)
	
	if( RC_BAD( rc = FlmAllocRandomGenerator( &gv_pSerialRandom)))
	{
		goto Exit;
	}

	gv_pSerialRandom->setSeed( (FLMUINT32)(uiTime ^ (FLMUINT)getpid()));
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void f_freeSerialNumberGenerator( void)
{
	if( (--gv_uiSerialInitCount) > 0)
	{
		return;
	}
	
	if( gv_pSerialRandom)
	{
		gv_pSerialRandom->Release();
		gv_pSerialRandom = NULL;
	}

	if( gv_hSerialMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hSerialMutex);
	}
}

/****************************************************************************
Desc:		This routine will use the operating system calls to generate a
			"globally unique" identifier.  Typically, this is based on the
			MAC address of an ethernet card installed in the machine.  If the
			machine does not have an ethernet card, or if the OS does not
			support generating GUIDs, this routine will generate a pseudo-GUID
			using a random number generator.  A serial number is 16-bytes.
****************************************************************************/
RCODE FLMAPI f_createSerialNumber(
	FLMBYTE *		pszSerialNum)
{
	RCODE						rc = NE_FLM_OK;

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

#elif defined( FLM_UNIX) || defined( FLM_NLM)

	// Generate a pseudo GUID value

	f_assert( gv_hSerialMutex != F_MUTEX_NULL);

	f_mutexLock( gv_hSerialMutex);

	UD2FBA( (FLMUINT32)gv_pSerialRandom->getUINT32(), &pszSerialNum[ 0]);
	UD2FBA( (FLMUINT32)gv_pSerialRandom->getUINT32(), &pszSerialNum[ 4]);
	UD2FBA( (FLMUINT32)gv_pSerialRandom->getUINT32(), &pszSerialNum[ 8]);
	UD2FBA( (FLMUINT32)gv_pSerialRandom->getUINT32(), &pszSerialNum[ 12]);

	f_mutexUnlock( gv_hSerialMutex);

#endif

#if defined( FLM_WIN)
Exit:
#endif

	return( rc);
}

/****************************************************************************
Desc: Generates a table of remainders for each 8-bit byte.  The resulting
		table is used by f_updateCRC to calculate a CRC value.  The table
		must be freed via a call to f_free.
*****************************************************************************/
FSTATIC RCODE f_initCRCTable(
	FLMUINT32 **	ppui32CRCTbl)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT32 *		pTable;
	FLMUINT32		ui32Val;
	FLMUINT32		ui32Loop;
	FLMUINT32		ui32SubLoop;

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
void FLMAPI f_updateCRC(
	const void *		pvBuffer,
	FLMUINT				uiCount,
	FLMUINT32 *			pui32CRC)
{
	FLMBYTE *			pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT32			ui32CRC = *pui32CRC;
	FLMUINT				uiLoop;

	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		ui32CRC = (ui32CRC >> 8) ^ gv_pui32CRCTbl[
			((FLMBYTE)(ui32CRC & 0x000000FF)) ^ pucBuffer[ uiLoop]];
	}

	*pui32CRC = ui32CRC;
}

/****************************************************************************
Desc: 	
****************************************************************************/
void FLMAPI f_getenv(
	const char *	pszKey,
	FLMBYTE *		pszBuffer,
	FLMUINT			uiBufferSize,
	FLMUINT *		puiValueLen)
{
	FLMUINT			uiValueLen = 0;

	if( !uiBufferSize)
	{
		goto Exit;
	}
	
	pszBuffer[ 0] = 0;
	
#if defined( FLM_WIN) || defined( FLM_UNIX)
	char *	pszValue;
	
   if( (pszValue = getenv( pszKey)) != NULL &&
		 (uiValueLen = f_strlen( pszValue)) < uiBufferSize)
	{
		f_strcpy( (char *)pszBuffer, pszValue);
	}
#else
	F_UNREFERENCED_PARM( pszKey);
#endif

Exit:

	if( puiValueLen)
	{
		*puiValueLen = uiValueLen;
	}

	return;
}

/***************************************************************************
Desc:		Sort an array of items
****************************************************************************/
void FLMAPI f_qsort(
	void *					pvBuffer,
	FLMUINT					uiLowerBounds,
	FLMUINT					uiUpperBounds,
	F_SORT_COMPARE_FUNC	fnCompare,
	F_SORT_SWAP_FUNC		fnSwap)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiCurrentPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	uiCurrentPos = uiMIDPos;

	for (;;)
	{
		while (uiLBPos == uiMIDPos ||
					((iCompare = 
						fnCompare( pvBuffer, uiLBPos, uiCurrentPos)) < 0))
		{
			if( uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while( uiUBPos == uiMIDPos ||
					(((iCompare = 
						fnCompare( pvBuffer, uiCurrentPos, uiUBPos)) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}
		
		if( uiLBPos < uiUBPos)
		{
			// Exchange [uiLBPos] with [uiUBPos].

			fnSwap( pvBuffer, uiLBPos, uiUBPos);
			uiLBPos++;
			uiUBPos--;
		}
		else
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		fnSwap( pvBuffer, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{
		// Exchange [uUBPos] with [uiMIDPos]

		fnSwap( pvBuffer, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos)
							? uiMIDPos - uiLowerBounds
							: 0;

	uiRightItems = (uiMIDPos + 1 < uiUpperBounds)
							? uiUpperBounds - uiMIDPos
							: 0;

	if( uiLeftItems < uiRightItems)
	{
		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if( uiLeftItems)
		{
			f_qsort( pvBuffer, uiLowerBounds, uiMIDPos - 1, fnCompare, fnSwap);
		}

		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems)
	{
		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			f_qsort( pvBuffer, uiMIDPos + 1, uiUpperBounds, fnCompare, fnSwap);
		}

		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_qsortUINTCompare(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	FLMUINT		uiLeft = *(((FLMUINT *)pvBuffer) + uiPos1);
	FLMUINT		uiRight = *(((FLMUINT *)pvBuffer) + uiPos2);

	if( uiLeft < uiRight)
	{
		return( -1);
	}
	else if( uiLeft > uiRight)
	{
		return( 1);
	}

	return( 0);
}

/***************************************************************************
Desc:
****************************************************************************/
void FLMAPI f_qsortUINTSwap(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	FLMUINT *	puiArray = (FLMUINT *)pvBuffer;
	FLMUINT		uiTmp = puiArray[ uiPos1];

	puiArray[ uiPos1] = puiArray[ uiPos2];
	puiArray[ uiPos2] = uiTmp;
}

/****************************************************************************
Desc:
****************************************************************************/
void * FLMAPI f_memcpy(
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
void * FLMAPI f_memmove(
	void *			pvDest,
	const void *	pvSrc,
	FLMSIZET			uiLength)
{
	return( memmove( pvDest, pvSrc, uiLength));
}

/****************************************************************************
Desc:
****************************************************************************/
void * FLMAPI f_memset(
	void *				pvDest,
	unsigned char		ucByte,
	FLMSIZET				uiLength)
{
	return( memset( pvDest, ucByte, uiLength));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_memcmp(
	const void *		pvMem1,
	const void *		pvMem2,
	FLMSIZET				uiLength)
{
	return( memcmp( pvMem1, pvMem2, uiLength));
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strcpy(
	char *			pszDest,
	const char *	pszSrc)
{
	return( strcpy( pszDest, pszSrc));
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strncpy(
	char *			pszDest,
	const char *	pszSrc,
	FLMSIZET			uiLength)
{
	return( strncpy( pszDest, pszSrc, uiLength));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI f_strlen(
	const char *	pszStr)
{
	return( strlen( pszStr));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_strcmp(
	const char *		pszStr1,
	const char *		pszStr2)
{
	return( strcmp( pszStr1, pszStr2));
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_stricmp(
	const char *		pszStr1,
	const char *		pszStr2)
{
#ifdef FLM_WIN
	return( _stricmp( pszStr1, pszStr2));
#else 
	while( f_toupper( *pszStr1) == f_toupper( *pszStr2) && *pszStr1)
	{
		pszStr1++;
		pszStr2++;
	}
	return( (FLMINT)( f_toupper( *pszStr1) - f_toupper( *pszStr2)));
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_strncmp(
	const char *		pszStr1,
	const char *		pszStr2,
	FLMSIZET				uiLength)
{
	return( strncmp( pszStr1, pszStr2, uiLength));
}
	
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_strnicmp(
	const char *		pszStr1,
	const char *		pszStr2,
	FLMSIZET				uiLength)
{
#ifdef FLM_WIN
	return( _strnicmp( pszStr1, pszStr2, uiLength));
#else
	FLMINT				iLen = (FLMINT)uiLength;

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

#endif
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strcat(
	char *				pszDest,
	const char *		pszSrc)
{
	return( strcat( pszDest, pszSrc));
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strncat(
	char *				pszDest,
	const char *		pszSrc,
	FLMSIZET				uiLength)
{
	return( strncat( pszDest, pszSrc, uiLength));
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strchr(
	const char *		pszStr,
	unsigned char		ucByte)
{
	return( strchr( pszStr, ucByte));
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strrchr(
	const char *		pszStr,
	unsigned char		ucByte)
{
	return( strrchr( pszStr, ucByte));
}

/****************************************************************************
Desc:
****************************************************************************/
char * FLMAPI f_strstr(
	const char *		pszStr1,
	const char *		pszStr2)
{
	return( strstr( pszStr1, pszStr2));
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
char * FLMAPI f_strupr(
	char *				pszStr)
{
	return( _strupr( pszStr));
}
#endif

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 FLMAPI f_atomicInc(
	FLMATOMIC *			piTarget)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)atomic_retadd( (unsigned long *)piTarget, 1));
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
		#error Atomic operations are not supported
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 FLMAPI f_atomicDec(
	FLMATOMIC *			piTarget)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)atomic_retadd( (unsigned long *)piTarget, -1));
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
		#error Atomic operations are not supported
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT32 FLMAPI f_atomicExchange(
	FLMATOMIC *			piTarget,
	FLMINT32				i32NewVal)
{
	#if defined( FLM_NLM)
	{
		return( (FLMINT32)atomic_xchg( (unsigned long *)piTarget, i32NewVal));
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
		#error Atomic operations are not supported
	#endif
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT FLMAPI F_Object::getRefCount( void)
{
	return( m_refCnt);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT FLMAPI F_Object::AddRef( void)
{
	return( ++m_refCnt);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMINT FLMAPI F_Object::Release( void)
{
	FLMINT		iRefCnt = --m_refCnt;

	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/**********************************************************************
Desc:
**********************************************************************/
IF_FileSystem * f_getFileSysPtr( void)
{
	return( gv_pFileSystem);
}

/**********************************************************************
Desc:
**********************************************************************/
IF_ThreadMgr * f_getThreadMgrPtr( void)
{
	return( gv_pThreadMgr);
}

/**********************************************************************
Desc:
**********************************************************************/
FLMUINT FLMAPI f_getMaxFileSize( void)
{
	return( gv_uiMaxFileSize);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FLMAPI f_readSEN(
	IF_IStream *	pIStream,
	FLMUINT *		puiValue,
	FLMUINT *		puiLength)
{
	RCODE				rc;
	FLMUINT64		ui64Tmp;

	if( RC_BAD( rc = f_readSEN64( pIStream, &ui64Tmp, puiLength)))
	{
		goto Exit;
	}

	if( ui64Tmp > ~((FLMUINT)0))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( puiValue)
	{
		*puiValue = (FLMUINT)ui64Tmp;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FLMAPI f_readSEN64(
	IF_IStream *		pIStream,
	FLMUINT64 *			pui64Value,
	FLMUINT *			puiLength)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiLen;
	FLMUINT				uiSENLength;
	FLMBYTE				ucBuffer[ 16];
	const FLMBYTE *	pucBuffer;

	uiLen = 1;
	if( RC_BAD( rc = pIStream->read( 
		(char *)&ucBuffer[ 0], uiLen, &uiLen)))
	{
		goto Exit;
	}

	uiSENLength = 	gv_ucSENLengthArray[ ucBuffer[ 0]];
	uiLen = uiSENLength - 1;

	if( puiLength)
	{
		*puiLength = uiSENLength;
	}

	if( pui64Value)
	{
		pucBuffer = &ucBuffer[ 1];
	}
	else
	{
		pucBuffer = NULL;
	}

	if( uiLen)
	{
		if( RC_BAD( rc = pIStream->read( 
			(char *)pucBuffer, uiLen, &uiLen)))
		{
			goto Exit;
		}
	}

	if( pui64Value)
	{
		pucBuffer = &ucBuffer[ 0];
		if( RC_BAD( rc = f_decodeSEN64( &pucBuffer,
			&ucBuffer[ sizeof( ucBuffer)], pui64Value)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
FLMUINT FLMAPI f_getSENLength(
	FLMBYTE 					ucByte)
{
	return( gv_ucSENLengthArray[ ucByte]);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI f_decodeSEN64(
	const FLMBYTE **		ppucBuffer,
	const FLMBYTE *		pucEnd,
	FLMUINT64 *				pui64Value)
{
	RCODE						rc = NE_FLM_OK;
	FLMUINT					uiSENLength;
	const FLMBYTE *		pucBuffer = *ppucBuffer;

	uiSENLength = gv_ucSENLengthArray[ *pucBuffer];
	if( pucBuffer + uiSENLength > pucEnd)
	{
		if (pui64Value)
		{
			*pui64Value = 0;
		}
		rc = RC_SET( NE_FLM_BAD_SEN);
		goto Exit;
	}

	if (pui64Value)
	{
		switch( uiSENLength)
		{
			case 1:
				*pui64Value = *pucBuffer;
				break;
	
			case 2:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x3F)) << 8) + pucBuffer[ 1];
				break;
	
			case 3:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x1F)) << 16) +
					(((FLMUINT64)pucBuffer[ 1]) << 8) + pucBuffer[ 2];
				break;
	
			case 4:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x0F)) << 24) +
					(((FLMUINT64)pucBuffer[ 1]) << 16) +
					(((FLMUINT64)pucBuffer[ 2]) << 8) + pucBuffer[ 3];
				break;
	
			case 5:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x07)) << 32) +
					(((FLMUINT64)pucBuffer[ 1]) << 24) +
					(((FLMUINT64)pucBuffer[ 2]) << 16) +
					(((FLMUINT64)pucBuffer[ 3]) << 8) + pucBuffer[ 4];
				break;
	
			case 6:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x03)) << 40) +
					(((FLMUINT64)pucBuffer[ 1]) << 32) +
					(((FLMUINT64)pucBuffer[ 2]) << 24) +
					(((FLMUINT64)pucBuffer[ 3]) << 16) +
					(((FLMUINT64)pucBuffer[ 4]) << 8) + pucBuffer[ 5];
				break;
	
			case 7:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x01)) << 48) +
					(((FLMUINT64)pucBuffer[ 1]) << 40) +
					(((FLMUINT64)pucBuffer[ 2]) << 32) +
					(((FLMUINT64)pucBuffer[ 3]) << 24) +
					(((FLMUINT64)pucBuffer[ 4]) << 16) +
					(((FLMUINT64)pucBuffer[ 5]) << 8) + pucBuffer[ 6];
				break;
	
			case 8:
				*pui64Value = (((FLMUINT64)pucBuffer[ 1]) << 48) +
					(((FLMUINT64)pucBuffer[ 2]) << 40) +
					(((FLMUINT64)pucBuffer[ 3]) << 32) +
					(((FLMUINT64)pucBuffer[ 4]) << 24) +
					(((FLMUINT64)pucBuffer[ 5]) << 16) +
					(((FLMUINT64)pucBuffer[ 6]) << 8) + pucBuffer[ 7];
				break;
	
			case 9:
				*pui64Value = (((FLMUINT64)pucBuffer[ 1]) << 56) +
					(((FLMUINT64)pucBuffer[ 2]) << 48) +
					(((FLMUINT64)pucBuffer[ 3]) << 40) +
					(((FLMUINT64)pucBuffer[ 4]) << 32) +
					(((FLMUINT64)pucBuffer[ 5]) << 24) +
					(((FLMUINT64)pucBuffer[ 6]) << 16) +
					(((FLMUINT64)pucBuffer[ 7]) << 8) + pucBuffer[ 8];
				break;
	
			default:
				*pui64Value = 0;
				flmAssert( 0);
				break;
		}
	}

Exit:

	*ppucBuffer = pucBuffer + uiSENLength;

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI f_decodeSEN(
	const FLMBYTE **		ppucBuffer,
	const FLMBYTE *		pucEnd,
	FLMUINT *				puiValue)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT64		ui64Value;
	
	if( RC_BAD( rc = f_decodeSEN64( ppucBuffer, pucEnd, &ui64Value)))
	{
		return( rc);
	}
	
	if( ui64Value > FLM_MAX_UINT)
	{
		return( RC_SET_AND_ASSERT( NE_FLM_CONV_NUM_OVERFLOW));
	}
	
	if( puiValue)
	{
		*puiValue = (FLMUINT)ui64Value;
	}

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE f_shiftRightRetByte(
	FLMUINT64	ui64Num,
	FLMBYTE		ucBits)
{
	return( ucBits < 64 ? (FLMBYTE)(ui64Num >> ucBits) : 0);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI f_getSENByteCount(
	FLMUINT64	ui64Num)
{
	FLMUINT		uiCount = 0;

	if( ui64Num < 0x80)
	{
		return( 1);
	}

	while( ui64Num)
	{
		uiCount++;
		ui64Num >>= 7;
	}

	// If the high bit is set, the counter will be incremented 1 beyond
	// the actual number of bytes need to represent the SEN.  We will need
	// to re-visit this if we ever go beyond 64-bits.

	return( uiCount < FLM_MAX_SEN_LEN ? uiCount : FLM_MAX_SEN_LEN);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
FLMUINT FLMAPI f_encodeSEN(
	FLMUINT64		ui64Value,
	FLMBYTE **		ppucBuffer,
	FLMUINT			uiSizeWanted)
{
	FLMBYTE *		pucBuffer = *ppucBuffer;
	FLMUINT			uiSenLen = f_getSENByteCount( ui64Value);

	flmAssert( uiSizeWanted <= FLM_MAX_SEN_LEN && 
				  (!uiSizeWanted || uiSizeWanted >= uiSenLen));

	uiSenLen = uiSizeWanted > uiSenLen ? uiSizeWanted : uiSenLen;

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	return( uiSenLen);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
RCODE FLMAPI f_encodeSEN(
	FLMUINT64		ui64Value,
	FLMBYTE **		ppucBuffer,
	FLMBYTE *		pucEnd)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucBuffer = *ppucBuffer;
	FLMUINT			uiSenLen = f_getSENByteCount( ui64Value);
	
	if( *ppucBuffer + uiSenLen > pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
FLMUINT FLMAPI f_encodeSENKnownLength(
	FLMUINT64		ui64Value,
	FLMUINT			uiSenLen,
	FLMBYTE **		ppucBuffer)
{
	FLMBYTE *			pucBuffer = *ppucBuffer;

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = f_shiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	return( uiSenLen);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI FlmGetXMLObject(
	IF_XML **				ppXmlObject)
{
	*ppXmlObject = gv_pXml;
	(*ppXmlObject)->AddRef();
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
IF_XML * f_getXmlObjPtr( void)
{
	return( gv_pXml);
}
