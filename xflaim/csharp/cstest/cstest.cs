//------------------------------------------------------------------------------
// Desc:	CSharp Tester
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
using xflaim;

namespace cstest
{
	public class Tester
	{
		private const string CREATE_DB_NAME = "create.db";
		private const string COPY_DB_NAME = "copy.db";
		private const string RENAME_DB_NAME = "rename.db";
		private const string RESTORE_DB_NAME = "restore.db";
		private const string REBUILD_DB_NAME = "rebuild.db";
		private const string BACKUP_PATH = "backup";

		//--------------------------------------------------------------------------
		// Begin a test.
		//--------------------------------------------------------------------------
		static void beginTest( 
			string	sTestName)
		{
			System.Console.Write( "{0} ...", sTestName);
		}

		//--------------------------------------------------------------------------
		// End a test.
		//--------------------------------------------------------------------------
		static void endTest(
			bool	bPassed)
		{
			if (bPassed)
			{
				System.Console.WriteLine( "PASS");
			}
			else
			{
				System.Console.WriteLine( "FAIL");
			}
		}

		//--------------------------------------------------------------------------
		// End a test with an exception
		//--------------------------------------------------------------------------
		static void endTest(
			XFlaimException	ex,
			string				sWhat)
		{
			endTest( false);
			System.Console.Write( "Error {0}: ", sWhat);
			if (ex.getRCode() == RCODE.NE_XFLM_OK)
			{
				System.Console.WriteLine( "{0}", ex.getString());
			}
			else
			{
				System.Console.WriteLine( "{0}", ex.getRCode());
			}
		}

		//--------------------------------------------------------------------------
		// Create database test
		//--------------------------------------------------------------------------
		static bool createDbTest(
			DbSystem	dbSystem)
		{
			Db			db = null;
			RCODE	rc;

			beginTest( "Create Database Test (" + CREATE_DB_NAME + ")");

			for (;;)
			{
				rc = RCODE.NE_XFLM_OK;
				try
				{
					CREATE_OPTS	createOpts = new CREATE_OPTS();

					createOpts.uiBlockSize = 8192;
					createOpts.uiVersionNum = (uint)DBVersions.XFLM_CURRENT_VERSION_NUM;
					createOpts.uiMinRflFileSize = 2000000;
					createOpts.uiMaxRflFileSize = 20000000;
					createOpts.bKeepRflFiles = 1;
					createOpts.bLogAbortedTransToRfl = 1;
					createOpts.uiDefaultLanguage = (uint)Languages.FLM_DE_LANG;
					db = dbSystem.dbCreate( CREATE_DB_NAME, null, null, null, null, createOpts);
				}
				catch (XFlaimException ex)
				{
					rc = ex.getRCode();

					if (rc != RCODE.NE_XFLM_FILE_EXISTS)
					{
						endTest( ex, "creating database");
						return( false);
					}
				}
				if (rc == RCODE.NE_XFLM_OK)
				{
					break;
				}

				// rc better be NE_XFLM_FILE_EXISTS - try to delete the file

				try
				{
					dbSystem.dbRemove( CREATE_DB_NAME, null, null, true);
				}
				catch (XFlaimException ex)
				{
					endTest( ex, "removing database");
					return( false);
				}
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			endTest( true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Open database test.
		//--------------------------------------------------------------------------
		static bool openDbTest(
			DbSystem	dbSystem)
		{
			Db	db = null;

			beginTest( "Open Database Test (" + CREATE_DB_NAME + ")");

			try
			{
				db = dbSystem.dbOpen( CREATE_DB_NAME, null, null, null, false);
			}
			catch (XFlaimException ex)
			{
				endTest( ex, "opening database");
				return( false);
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			endTest( true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Copy database test.
		//--------------------------------------------------------------------------
		static bool copyDbTest(
			DbSystem	dbSystem)
		{

			// Try copying the database

			MyDbCopyStatus	copyStatus = new MyDbCopyStatus();

			beginTest( "Copy Database Test (" + CREATE_DB_NAME + " --> " + COPY_DB_NAME + ")");
			try
			{
				dbSystem.dbCopy( CREATE_DB_NAME, null, null, COPY_DB_NAME, null, null, copyStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( ex, "copying database");
				return( false);
			}
			endTest( true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Remove database test.
		//--------------------------------------------------------------------------
		static bool removeDbTest(
			DbSystem	dbSystem,
			string	sDbName)
		{
			beginTest( "Remove Database Test (" + sDbName + ")");
			try
			{
				dbSystem.dbRemove( sDbName, null, null, true);
			}
			catch (XFlaimException ex)
			{
				endTest( ex, "removing database");
				return( false);
			}
			endTest( true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Main for tester program
		//--------------------------------------------------------------------------
		static void Main()
		{
			DbSystem dbSystem = new DbSystem();

			// Database create test

			if (!createDbTest( dbSystem))
			{
				return;
			}

			// Database open test

			if (!openDbTest( dbSystem))
			{
				return;
			}

			// Database copy test

			if (!copyDbTest( dbSystem))
			{
				return;
			}

			// Database remove test

			if (!removeDbTest( dbSystem, CREATE_DB_NAME))
			{
				return;
			}
			if (!removeDbTest( dbSystem, COPY_DB_NAME))
			{
				return;
			}
		}
	}

	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public class MyDbCopyStatus : DbCopyStatus
	{
		public RCODE dbCopyStatus(
			ulong			uiBytesToCopy,
			ulong			uiBytesCopied,
			string		sSrcFileName,
			string		sDestFileName)
		{
			if (sSrcFileName != null)
			{
				System.Console.WriteLine( "\nSrc File: {0}, Dest File {1}", sSrcFileName, sDestFileName);
			}
			System.Console.Write( "Bytes To Copy: {0}, Bytes Copied: {1}\r", uiBytesToCopy, uiBytesCopied);
			return( RCODE.NE_XFLM_OK);
		}
	}
}

