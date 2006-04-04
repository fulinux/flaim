//------------------------------------------------------------------------------
// Desc:	This include file contains the definition for the F_DbSystemFactory
//			class, which is used to create an interface to the F_DbSystem object.
//			The factory is only used by COM clients.
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
// $Id: fcomfact.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FCOMFACT_H
#define FCOMFACT_H

	class F_DbSystemFactory : public XFLMIClassFactory
	{
	public:
	
		RCODE XFLMAPI QueryInterface(
			RXFLMIID				riid,
			void **				ppvInt);

		RCODE XFLMAPI CreateInstance(
			XFLMIUnknown *		pUnkOut,
			RXFLMIID				riid,
			void **				ppvOut);

		RCODE XFLMAPI LockServer(
			bool					bLock);
			
		FLMINT XFLMAPI AddRef( void);

		FLMINT XFLMAPI Release( void);
	};

#endif // FCOMFACT_H

