//------------------------------------------------------------------------------
// Desc:	Positionable Input Stream
//
// Tabs:	3
//
//		Copyright (c) 2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: PosIStream.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * The PosIStream class provides a number of methods that allow java 
 * applications to access the XFlaim native environment, specifically, the
 * IF_PosIStream interface.
 */
public class PosIStream 
{
	PosIStream( 
		long			lRef,
		String 		sBuffer,
		DbSystem 	dbSystem) throws XFlaimException
	{
		if (lRef == 0)
		{
			throw new XFlaimException( -1, "No legal reference to an F_PosIStream");
		}
		
		m_this = lRef;
		
		if (dbSystem==null)
		{
			throw new XFlaimException( -1, "No legal dbSystem reference");
		}
		
		m_dbSystem = dbSystem;
		
		if (sBuffer == null)
		{
			throw new XFlaimException( -1, "No legal reference to a buffer");
		}
		
		m_sBuffer = sBuffer;
	}
	
	PosIStream( 
		long 			lRef,
		DbSystem 	dbSystem) throws XFlaimException
	{
		if (lRef == 0)
		{
			throw new XFlaimException( -1, "No legal reference to an IF_PosIStream");
		}
		
		m_this = lRef;
		
		if (dbSystem==null)
		{
			throw new XFlaimException( -1, "No legal dbSystem reference");
		}
		
		m_dbSystem = dbSystem;
		m_sBuffer = null;
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
	 * Desc:
	 */
	private native void _release( long iThis);

	private long			m_this;
	private DbSystem		m_dbSystem;
	private String			m_sBuffer;
}
