//-------------------------------------------------------------------------
// Desc:	Module loading
// Tabs:	3
//
//		Copyright (c) 2004-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftkmod.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaim.h"

#ifndef FLM_OSX

// OS X 10.2 and earlier don't have dlopen, dlsym, etc...  They have a totally
// different mechanism and since these functions are only used for our unit
// tests,  it's probably not worth the time to sort it all out.  In 10.3,
// the dl functions were implemented in terms of the native functions and
// this should all work.  Pity there's no easy way to determine which version
// we're on...


#include "flaimsys.h"

#ifdef FLM_UNIX
	#include <dlfcn.h>
#endif


#ifdef FLM_WIN

	FSTATIC RCODE flmSymLoadWin( 
		const FLMBYTE * pszName, 
		FlmModHandle hMod,  
		void ** ppvSym);

#elif defined (FLM_NLM)

	FSTATIC RCODE flmSymLoadNW( 
		const FLMBYTE * pszName, 
		void ** ppvSym);

#elif defined( FLM_UNIX)

	FSTATIC RCODE flmSymLoadUnix( 
		const FLMBYTE * pszName, 
		FlmModHandle hMod,  
		void ** ppvSym);

#endif

/****************************************************************************
Desc:	This routine loads a shared library (cross-platform)
****************************************************************************/
RCODE FlmModLoad( 
	const char *		pszName,
	FlmModHandle * 	phMod)
{
	RCODE		rc = FERR_OK;
#ifdef FLM_WIN
   HMODULE	hMod;

   if ((hMod = LoadLibrary( pszName)) == NULL)
   {
      rc = MapWinErrorToFlaim( GetLastError(), FERR_LOAD_LIBRARY);
		goto Exit;
	}

	if ( phMod)
	{
      *phMod = (FlmModHandle)hMod;
	}

#elif defined (FLM_UNIX)	
	FlmModHandle hMod;

	if (!(hMod = dlopen( (const char *)pszName, RTLD_LAZY)))
   {
		// Note, I can't seem to get any real diagnostic info back
		// from the operating system. It does not set errno on failure.
      rc = RC_SET( FERR_LOAD_LIBRARY);
		goto Exit;
   }

	if ( phMod)
	{
		*phMod = hMod; 
	}

#elif defined (FLM_NLM)

	char				szLoadName[ 128];
	LONG				lErr;

	// The module handle will be passed back in the first 4 bytes of the name
	f_strcpy( &szLoadName[4], pszName);
	
	if ( ( lErr = LoadModule( 
		GetSystemConsoleScreen(), 
		(unsigned char *)&szLoadName[4], 
		LO_RETURN_HANDLE)) != 0)
	{
		rc = RC_SET( FERR_LOAD_LIBRARY);
		goto Exit;
	}

	*phMod = (FlmModHandle)&szLoadName[0];

#else
	#error Unsupported OS
#endif

Exit:
	return( rc);
}

#ifndef FLM_NLM
/****************************************************************************
Desc:	This routine unloads a shared library (cross-platform)
****************************************************************************/
RCODE FlmModUnload( FlmModHandle * phMod)
{
	RCODE rc = FERR_OK;

	flmAssert( phMod);
#ifdef FLM_WIN

	if (!FreeLibrary((HMODULE)*phMod))
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_UNLOAD_LIBRARY);
		goto Exit;
	}
   *phMod = NULL;

#elif defined (FLM_UNIX)
	if (dlclose(phMod) != 0)
	{
		rc = RC_SET( FERR_UNLOAD_LIBRARY);
		goto Exit;
	}
	*phMod = NULL;
#else
	#error Unsupported OS
#endif
Exit:
	return( rc);
}
#endif

#ifdef FLM_NLM

/****************************************************************************
Desc:	This routine creates a FLAIM file. For whatever reason, NetWare unloads 
a module using its path and not its handle. While routines exist that look up 
the path given the module handle, I can't use them because they are all LibC 
calls
****************************************************************************/
RCODE FlmModUnload( FLMBYTE * pszModPath)
{
	int		iErr;
	RCODE		rc = FERR_OK;

	if ( (iErr = UnloadModule( 
		NULL, 
		(const char *)pszModPath)) != 0)
	{
		rc = MapNWtoFlaimError( iErr, FERR_UNLOAD_LIBRARY);
		goto Exit;
	}

Exit:
	return( rc);
}
#endif

#ifdef FLM_WIN

/****************************************************************************
Desc:	This routine loads a symbol from a .dll file
****************************************************************************/
FSTATIC RCODE flmSymLoadWin( 
	const FLMBYTE * pszName, 
	FlmModHandle hMod,  
	void ** ppvSym)
{
	RCODE		rc = FERR_OK;
	void *	pvSym;

	if ( ( pvSym = GetProcAddress( (HMODULE)hMod, (LPCSTR)pszName)) == NULL)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_IMPORT_SYMBOL);
		goto Exit;
	}

Exit:
	if ( ppvSym && RC_OK( rc))
	{
		*ppvSym = pvSym;
	}

	return( rc);
}

#elif defined (FLM_NLM)

/****************************************************************************
Desc:	This routine loads a symbol from a .nlm file
****************************************************************************/
FSTATIC RCODE flmSymLoadNW( 
	const FLMBYTE * pszName,
	void ** ppvSym)
{
	RCODE			rc = FERR_OK;
	void *		pvSym;
	FLMUINT		uiNameLen;
	FLMBYTE *	pszLengthPlusName = NULL;

	// NetWare is a little different (of course). You use your own handle 
	// to query for and retrieve symbols. All loaded modules are queried.

	uiNameLen = f_strlen( (const char *)pszName);

	if ( uiNameLen > 255)
	{

		// If the name's length can't be represented in a single byte,
		// it's too large.

		rc = RC_SET( FERR_VALUE_TOO_LARGE);
		goto Exit;
	}

	if ( RC_BAD( rc = f_alloc( uiNameLen + 2, &pszLengthPlusName)))
	{
		goto Exit;
	}

	// The first byte of the name should be the length of the name

	pszLengthPlusName[0] = (FLMBYTE)uiNameLen;
	f_strcpy( (char *)&pszLengthPlusName[1], (const char *)pszName);
  
   if (( pvSym = (void *)ImportPublicSymbol( 
		f_getNLMHandle(),  
		(BYTE*)pszLengthPlusName)) == NULL)
	{
		rc = RC_SET( FERR_IMPORT_SYMBOL);
		goto Exit;
	}

Exit:
	if ( pszLengthPlusName)
	{
		f_free( &pszLengthPlusName);
	}

	if ( ppvSym && RC_OK( rc))
	{
		*ppvSym = pvSym;
	}
	return( rc);
}

#elif defined( FLM_UNIX)

/****************************************************************************
Desc:	This routine loads a symbol from a .so file.
****************************************************************************/
FSTATIC RCODE flmSymLoadUnix( 
	const FLMBYTE * pszName, 
	FlmModHandle hMod,  
	void ** ppvSym)
{
	RCODE			rc = FERR_OK;
	void *		pvSym;

	if ( ( pvSym = dlsym( hMod, (const char*)pszName)) == NULL)
	{
		rc = RC_SET( FERR_IMPORT_SYMBOL);
		goto Exit;
	}

Exit:
	if ( ppvSym && RC_OK( rc))
	{
		*ppvSym = pvSym;
	}

	return( rc);
}

#endif

/****************************************************************************
Desc:	This routine loads a symbol from a shared library (cross-platform)
****************************************************************************/
RCODE FlmSymLoad( const FLMBYTE * pszName, FlmModHandle hMod,  void ** ppvSym)
{
#ifdef FLM_WIN
	return flmSymLoadWin( pszName, hMod, ppvSym);
#elif defined (FLM_NLM)
	F_UNREFERENCED_PARM( hMod);
	return flmSymLoadNW( pszName, ppvSym);
#elif defined (FLM_UNIX)
	return flmSymLoadUnix( pszName, hMod, ppvSym);
#endif
}

/****************************************************************************
Desc:	This routine unloads a shared library symbol (cross-platform, but only 
really useful on NetWare since it seems to be the only platform that keeps a 
reference count on loaded symbols)
****************************************************************************/
RCODE FlmSymUnload(
  const FLMBYTE * pszName)
{
#ifdef FLM_NLM
	RCODE			rc = FERR_OK;
	FLMUINT		uiSymLen;
	FLMBYTE *	pszLenAndSymbol = NULL;

	uiSymLen = f_strlen( (const char *)pszName);

	if ( uiSymLen > 255)
	{
		rc = RC_SET( FERR_VALUE_TOO_LARGE);
		goto Exit;
	}

	if ( RC_BAD( rc = f_alloc( uiSymLen + 2, &pszLenAndSymbol)))
	{
		goto Exit;
	}

	pszLenAndSymbol[0] = (FLMBYTE)uiSymLen;
	f_strcpy( (char *)&pszLenAndSymbol[1], (const char *)pszName);

	if ((UnImportPublicSymbol (
		f_getNLMHandle(), pszLenAndSymbol)) != 0)
	{
		rc = RC_SET( FERR_UNIMPORT_SYMBOL);
		goto Exit;
	}

Exit:
	if( pszLenAndSymbol)
	{
		f_free( &pszLenAndSymbol);
	}

	return( rc);
#else

	// This is only really necessary on NetWare

	F_UNREFERENCED_PARM( pszName);
	return FERR_OK;
#endif
}

#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_OSX)
void gv_fnlm()
{
}
#endif
