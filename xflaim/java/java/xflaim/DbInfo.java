//------------------------------------------------------------------------------
// Desc:	Db Copy Status
//
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
// $Id: DbInfo.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * To change the template for this generated type comment go to
 * Window&gt;Preferences&gt;Java&gt;Code Generation&gt;Code and Comments
 */
public class DbInfo 
{
	DbInfo(
		long		lRef) throws XFlaimException
	{
		if (lRef == 0)
		{
			throw new XFlaimException( -1, "No legal reference");
		}
		
		m_this = lRef;
	}

	/**
	 * Desc:
	 */
	protected void finalize()
	{
		_release( m_this);
	}

	/**
	 * Desc:
	 */
	private native void _release(
		long		lThis);
		
	private long	m_this;
}
