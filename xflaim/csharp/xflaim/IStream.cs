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

using System;
using System.Runtime.InteropServices;

namespace xflaim
{

	/// <summary>
	/// The IStream class encapsulates an IF_IStream object that was allocated
	/// in unmanaged space.  It will make sure that the stream object will be
	/// freed when the C# IStream object goes away.
	/// NOTE: This object should NEVER be allocated by an application directly.
	/// It is returned from various methods in the <see cref="DbSystem"/> class,
	/// such as <see cref="DbSystem.openBufferIStream"/>,
	/// <see cref="DbSystem.openFileIStream"/>, etc.
	/// </summary>
	public class IStream 
	{
		private ulong			m_pIStream;		// Pointer to IF_IStream object allocated in unmanaged space.
		private DbSystem		m_dbSystem;

		/// <summary>
		/// Constructor.
		/// </summary>
		/// <param name="pIStream">
		/// Pointer to IF_IStream object that was allocated from unmanaged space.
		/// </param>
		/// <param name="dbSystem">
		/// Pointer to <see cref="DbSystem"/> object.
		/// </param>
		internal IStream(
			ulong			pIStream,
			DbSystem 	dbSystem)
		{
			if (pIStream == 0)
			{
				throw new XFlaimException( "Invalid IF_IStream object pointer");
			}
			
			m_pIStream = pIStream;
			
			if (dbSystem == null)
			{
				throw new XFlaimException( "Invalid DbSystem object");
			}
			
			m_dbSystem = dbSystem;

			// Must call something inside of DbSystem.  Otherwise, the
			// m_dbSystem object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_dbSystem.getDbSystem() == 0)
			{
				throw new XFlaimException( "Invalid DbSystem.IF_DbSystem object");
			}
		}
		
		/// <summary>
		/// Destructor
		/// </summary>
		~IStream()
		{
			if (m_pIStream != 0)
			{
				xflaim_IStream_Release( m_pIStream);
				m_pIStream = 0;
			}

			m_dbSystem = null;
		}

		[DllImport("xflaim")]
		private static extern void xflaim_IStream_Release(
			ulong	pIStream);

		internal ulong getIStream()
		{
			return( m_pIStream);
		}
	}
}
