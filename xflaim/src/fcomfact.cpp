//------------------------------------------------------------------------------
// Desc:	This file contains the implementation for the F_DbSystemFactory
//			class.
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcomfact.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#define PCOM_INIT_GUID
#include "flaimsys.h"
#include "fcomfact.h"

/********************************************************************
Desc: 
*********************************************************************/
RCODE F_DbSystemFactory::QueryInterface(
	RXFLMIID		riid,
	void **		ppvInt)
{
	RCODE			rc = NE_XFLM_OK;
	
	if( (f_memcmp(&riid, &Internal_IID_XFLMIClassFactory,
				   sizeof( Internal_IID_XFLMIClassFactory)) == 0)	||
		 (f_memcmp(&riid, &Internal_IID_XFLMIUnknown,
				   sizeof( Internal_IID_XFLMIUnknown)) == 0) )
	{
		*ppvInt = this;
		AddRef();
	}
	else
	{
		rc = RC_SET( NE_XFLM_UNSUPPORTED_INTERFACE);
	}
	
	return( rc);
}

/********************************************************************
Desc: 
*********************************************************************/
FLMUINT32 F_DbSystemFactory::AddRef(void)
{
	LockModule();
	return 2;
}

/********************************************************************
Desc: 
*********************************************************************/
FLMUINT32 F_DbSystemFactory::Release(void)
{
	UnlockModule();
	return 1;
}

/********************************************************************
Desc: 
*********************************************************************/
RCODE F_DbSystemFactory::LockServer(
	bool		bLock)
{
	if( bLock)
	{
		LockModule();
	}
	else
	{
		UnlockModule();
	}

	return( NE_XFLM_OK);
}

/********************************************************************
Desc: 
*********************************************************************/
RCODE F_DbSystemFactory::CreateInstance(
	XFLMIUnknown *	pUnkOut,
	RXFLMIID			riid,
	void **			ppvOut)
{
	RCODE				rc = NE_XFLM_OK;
	F_DbSystem *	pDbSystem = NULL;

	if (pUnkOut)
	{
		rc = RC_SET( NE_XFLM_UNSUPPORTED_INTERFACE);
		goto Exit;
	}

	if( (pDbSystem = f_new F_DbSystem) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pDbSystem->QueryInterface( riid, ppvOut)))
	{
		goto Exit;
	}

Exit:

	if (pDbSystem)
	{
		pDbSystem->Release();
	}

	return( rc);
}
