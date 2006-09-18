//------------------------------------------------------------------------------
// Desc:	Db Class
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

	/// <remarks>
	/// The Db class provides a number of methods that allow C#
	/// applications to access an XFLAIM database.  A Db object
	/// is obtained by calling <see cref="DbSystem.dbCreate"/> or
	/// <see cref="DbSystem.dbOpen"/>
	/// </remarks>
	public class Db
	{

		/// <summary>
		/// Db constructor.
		/// </summary>
		/// <param name="cs_dbRef">
		/// Reference to an IF_Db object.
		/// </param>
		/// <param name="dbSystem">
		/// DbSystem object that this Db object is associated with.
		/// </param>
		public Db(
			ulong		cs_dbRef,
			DbSystem	dbSystem)
		{
			if (cs_dbRef == 0)
			{
				throw new XFlaimException( "Invalid IF_Db reference");
			}
			
			m_this = cs_dbRef;

			if (dbSystem == null)
			{
				throw new XFlaimException( "Invalid DbSystem reference");
			}
			
			m_dbSystem = dbSystem;
			
			// Must call something inside of DbSystem.  Otherwise, the
			// m_dbSystem object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_dbSystem.getRef() == 0)
			{
				throw new XFlaimException( "Invalid DbSystem.getRef()");
			}
		}

		/// <summary>
		/// Destructor.
		/// </summary>
		~Db()
		{
			close();
		}

		/// <summary>
		/// Close this database.
		/// </summary>
		public void close()
		{
			// Release the native pDb!
		
			if (m_this != 0)
			{
				xflaim_Db_Release( m_this);
				m_this = 0;
			}
		
			// Remove our reference to the dbSystem so it can be released.
		
			m_dbSystem = null;
		}

		// PRIVATE METHODS THAT ARE IMPLEMENTED IN C AND C++

		[DllImport("xflaim")]
		private static extern int xflaim_Db_Release(
			ulong	ui64This);

		/// <summary>
		/// Reference to C++ IF_Db object.
		/// </summary>
		public ulong 		m_this;
		private DbSystem 	m_dbSystem;
	}
}
