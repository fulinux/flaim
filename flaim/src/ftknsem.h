//-------------------------------------------------------------------------
// Desc:	Cross-platform toolkit for named semaphores - definitions.
// Tabs:	3
//
//		Copyright (c) 2001-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftknsem.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

// Named semaphores are different from regular semaphores in that they're 
// designed to control access to a resource in use by multiple processes.
// Regular semaphores might actually do this (depending on the platform) but
// without a name for each semaphore, there's no way for one process to
// signal the semaphore created by another process...

// This code exists primarily because of the need to coordinate the actions
// of DS and the HTTPCTL utility on Windows and Unix.  (Things are done
// differently under Netware.)  I never could get it to work properly under
// Netware, and since it's not needed there, I've commented all of the
// Netware stuff out.  The problem was so much that I couldn't get it
// to work, but that if the named semaphore was deleted at the wrong time,
// programs using it would cause errors, or worse, crash the system.  To
// make matters worse, I couldn't figure out a way to guarentee that the
// semaphore wouldn't be deleted at the wrong time.  In light of all this,
// I figured it was best if this class simply didn't exist on Netware.

#ifndef FTKNSEM_H
#define FTKNSEM_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

enum eNamedSemFlags { OpenOnly, CreateOnly, OpenOrCreate };

#if defined( FLM_WIN)
	typedef HANDLE					NAMEDSEMHANDLE;
#elif defined( FLM_UNIX)
	typedef int						NAMEDSEMHANDLE;
#elif defined( FLM_NLM)
#else
	#error "Unsupported Platform"
#endif

#if defined( FLM_WIN) || defined( FLM_UNIX)
class F_NamedSemaphore : public F_Base
{
public:
	F_NamedSemaphore(
		const char	*		pszName,
		FLMUINT				uiMaxCount = 1,
		eNamedSemFlags		eFlags = OpenOrCreate);

	virtual ~F_NamedSemaphore();

	RCODE wait( void);						
	
	RCODE signal( 
		FLMUINT			uiCount = 1);

	RCODE destroy( void);
	
	FINLINE FLMBOOL isInitialized( void) 
	{
		return m_bInitialized;
	}

private:

#if defined( FLM_UNIX)
	FLMUINT32 NameToUnixKey(
		const char *	pszName);
#endif

	NAMEDSEMHANDLE		m_hSem;
	FLMBOOL				m_bInitialized;
};
#endif

#include "fpackoff.h"

#endif	// FTKNSEM_H
