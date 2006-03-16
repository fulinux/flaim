//-------------------------------------------------------------------------
// Desc:	Definitions for mutex handling class
// Tabs:	3
//
//		Copyright (c) 1998-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fmutxref.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FMUTXREF_H
#define FMUTXREF_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/****************************************************************************
Desc:	This object provides management and use of a mutex
****************************************************************************/
class F_MutexRef : public F_Base
{
public:
	F_MutexRef(								// Constructor
		F_MUTEX * phMutex)			// Actual semaphore object to use.
	{
		m_phMutex = phMutex;
		m_uiLockCount = 0;
	};

	virtual ~F_MutexRef()					// Destructor - make sure lock count == 0.
	{
		/* Unlock the semaphore if it is locked. 

			The semaphore could be still locked on an error condition within the
			users code. 
		*/ 
		if( m_uiLockCount != 0)
		{
			(void)f_mutexUnlock( *m_phMutex);
		}
	};

	void Lock()								// Locks the semaphore
	{
		if( m_phMutex)
		{
			if( !m_uiLockCount)
			{
				(void)f_mutexLock( *m_phMutex);
			}
			m_uiLockCount++;
		}
	};

	void Unlock()							// Unlock the semaphore
	{
		if( m_phMutex)
		{
			//flmAssert( m_uiLockCount != 0);	// Should not be zero
			if( !( --m_uiLockCount))
			{
				(void)f_mutexUnlock( *m_phMutex);
			}
		}
	};

private:

	FLMUINT				m_uiLockCount;	// Number of times the semaphore has
												// been locked by the thread.
	F_MUTEX *			m_phMutex;			// Pointer to semaphore.  If NULL,
												// there is no need to lock and
												// unlock the semaphore because the
												// objects it is controlling are NOT
												// shared.
};

#include "fpackoff.h"

#endif
