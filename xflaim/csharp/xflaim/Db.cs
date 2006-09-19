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
		/// <param name="pDb">
		/// Reference to an IF_Db object.
		/// </param>
		/// <param name="dbSystem">
		/// DbSystem object that this Db object is associated with.
		/// </param>
		public Db(
			ulong		pDb,
			DbSystem	dbSystem)
		{
			if (pDb == 0)
			{
				throw new XFlaimException( "Invalid IF_Db reference");
			}
			
			m_pDb = pDb;

			if (dbSystem == null)
			{
				throw new XFlaimException( "Invalid DbSystem reference");
			}
			
			m_dbSystem = dbSystem;
			
			// Must call something inside of DbSystem.  Otherwise, the
			// m_dbSystem object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_dbSystem.getDbSystem() == 0)
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
		/// Return the pointer to the IF_Db object.
		/// </summary>
		/// <returns>Returns a pointer to the IF_Db object.</returns>
		public ulong getDb()
		{
			return( m_pDb);
		}

		/// <summary>
		/// Close this database.
		/// </summary>
		public void close()
		{
			// Release the native pDb!
		
			if (m_pDb != 0)
			{
				xflaim_Db_Release( m_pDb);
				m_pDb = 0;
			}
		
			// Remove our reference to the dbSystem so it can be released.
		
			m_dbSystem = null;
		}

		/// <summary>
		/// Sets up a backup operation.
		/// </summary>
		/// <param name="bFullBackup">
		/// Specifies whether the backup is to be a full backup (true) or an incremental backup (false).
		/// </param>
		/// <param name="bLockDb">
		/// Specifies whether the database should be locked during the back (a "warm" backup)
		/// or unlocked (a "hot" backup).
		/// </param>
		/// <param name="uiMaxLockWait">
		/// This parameter is only used if the bLockDb parameter is true.  It specifies the maximum
		/// number of seconds to wait to obtain a lock.
		/// </param>
		/// <returns>
		/// If successful, this method returns a <see cref="Backup"/> object which can then be used
		/// to perform the backup operation.  The database will be locked if bLockDb was specified.
		/// Otherwise, a read transaction will have been started to perform the backup.
		/// </returns>
		public Backup backupBegin(
			bool	bFullBackup,
			bool	bLockDb,
			uint	uiMaxLockWait)
		{
			int		rc = 0;
			ulong		pBackup;

			if ((rc = xflaim_Db_backupBegin( m_pDb, (bFullBackup ? 1 : 0),
				(bLockDb ? 1 : 0), uiMaxLockWait, out pBackup)) != 0)
			{
				throw new XFlaimException( rc);
			}

			return( new Backup( pBackup, this));
		}

		// PRIVATE METHODS THAT ARE IMPLEMENTED IN C AND C++

		[DllImport("xflaim")]
		private static extern void xflaim_Db_Release(
			ulong	pDb);

		[DllImport("xflaim")]
		private static extern int xflaim_Db_backupBegin(
			ulong			pDb,
			int			bFullBackup,
			int			bLockDb,
			uint			uiMaxLockWait,
			out ulong	ulBackupRef);

		/// <summary>
		/// Reference to C++ IF_Db object.
		/// </summary>
		public ulong 		m_pDb;
		private DbSystem 	m_dbSystem;
	}
}
