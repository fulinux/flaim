//------------------------------------------------------------------------------
// Desc:	Input Stream
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

package xflaim;

/**
 * The IStream class provides a number of methods that allow java 
 * applications to access the XFlaim native environment, specifically, the
 * IF_IStream interface.
 */
public class IStream 
{
	IStream( 
		long			lRef,
		DbSystem 	dbSystem) throws XFlaimException
	{
		if (lRef == 0)
		{
			throw new XFlaimException( -1, "No legal reference to an F_IStream");
		}
		
		m_this = lRef;
		
		if (dbSystem==null)
		{
			throw new XFlaimException( -1, "No legal dbSystem reference");
		}
		
		m_dbSystem = dbSystem;
	}
	
	/**
	 * Finalizer method used to release native resources on garbage collection.
	 */	
	public void finalize()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}

		m_dbSystem = null;
	}

	/**
	 *
	 */	
	public void release()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}

		m_dbSystem = null;
	}
	
	public long getThis()
	{
		return( m_this);
	}
	
	/**
	 *
	 */
	private native void _release( long iThis);

	private long			m_this;
	private DbSystem		m_dbSystem;
}

