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
		private const string BACKUP_PATH = "backup";
		private const string REBUILD_DB_NAME = "rebuild.db";

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
			bool	bWriteLine,
			bool	bPassed)
		{
			if (bWriteLine)
			{
				System.Console.Write( "\n");
			}
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
			bool					bWriteLine,
			XFlaimException	ex,
			string				sWhat)
		{
			endTest( bWriteLine, false);
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
						endTest( false, ex, "creating database");
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
					endTest( false, ex, "removing database");
					return( false);
				}
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			endTest( false, true);
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
				endTest( false, ex, "opening database");
				return( false);
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			endTest( false, true);
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
				endTest( copyStatus.outputLines(), ex, "copying database");
				return( false);
			}
			endTest( copyStatus.outputLines(), true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Backup database test.
		//--------------------------------------------------------------------------
		static bool backupDbTest(
			DbSystem	dbSystem)
		{
			Db					db = null;
			Backup			backup = null;
			MyBackupStatus	backupStatus = null;
			uint				uiSeqNum;

			// Try backing up the database

			beginTest( "Backup Database Test (" + COPY_DB_NAME + " to directory \"" + BACKUP_PATH + "\")");

			try
			{
				db = dbSystem.dbOpen( COPY_DB_NAME, null, null, null, false);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "opening database");
				return( false);
			}

			// Backup the database

			try
			{
				backup = db.backupBegin( true, false, 0);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling backupBegin");
				return( false);
			}

			// Perform the backup

			backupStatus = new MyBackupStatus();
			try
			{
				uiSeqNum = backup.backup( BACKUP_PATH, null, null, backupStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( backupStatus.outputLines(), ex, "calling backup");
				return( false);
			}

			// End the backup

			try
			{
				backup.endBackup();
			}
			catch (XFlaimException ex)
			{
				endTest( backupStatus.outputLines(), ex, "calling endBackup");
				return( false);
			}

			db.close();
			db = null;
			endTest( backupStatus.outputLines(), true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Restore database test.
		//--------------------------------------------------------------------------
		static bool restoreDbTest(
			DbSystem	dbSystem)
		{
			MyRestoreStatus	restoreStatus = null;

			// Try restoring the database

			beginTest( "Restore Database Test (from directory \"" + BACKUP_PATH + "\" to " + RESTORE_DB_NAME + ")");

			restoreStatus = new MyRestoreStatus();
			try
			{
				dbSystem.dbRestore( RESTORE_DB_NAME, null, null, BACKUP_PATH, null,
										null, restoreStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( restoreStatus.outputLines(), ex, "restoring database");
				return( false);
			}

			endTest( restoreStatus.outputLines(), true);
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
				endTest( false, ex, "removing database");
				return( false);
			}
			endTest( false, true);
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

			// Database backup test

			if (!backupDbTest( dbSystem))
			{
				return;
			}

			// Database restore test

			if (!restoreDbTest( dbSystem))
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
			if (!removeDbTest( dbSystem, RESTORE_DB_NAME))
			{
				return;
			}
		}
	}

	public class MyDbCopyStatus : DbCopyStatus
	{
		public MyDbCopyStatus()
		{
			m_bOutputLines = false;
		}

		public RCODE dbCopyStatus(
			ulong			ulBytesToCopy,
			ulong			ulBytesCopied,
			string		sSrcFileName,
			string		sDestFileName)
		{
			if (sSrcFileName != null)
			{
				System.Console.WriteLine( "\nSrc File: {0}, Dest File {1}", sSrcFileName, sDestFileName);
			}
			System.Console.Write( "Bytes To Copy: {0}, Bytes Copied: {1}\r", ulBytesToCopy, ulBytesCopied);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public bool outputLines()
		{
			return( m_bOutputLines);
		}

		private bool	m_bOutputLines;
	}

	public class MyBackupStatus : BackupStatus
	{
		public MyBackupStatus()
		{
			System.Console.WriteLine( " ");
			m_bOutputLines = false;
		}

		public RCODE backupStatus(
			ulong			ulBytesToDo,
			ulong			ulBytesDone)
		{
			System.Console.Write( "Bytes To Backup: {0}, Bytes Backed Up: {1}\r", ulBytesToDo, ulBytesDone);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public bool outputLines()
		{
			return( m_bOutputLines);
		}

		private bool	m_bOutputLines;
	}

	public class MyRestoreStatus : RestoreStatus
	{
		private ulong	m_ulNumTransCommitted;
		private ulong	m_ulNumTransAborted;
		private bool	m_bOutputLines;
		
		public MyRestoreStatus()
		{
			m_ulNumTransCommitted = 0;
			m_ulNumTransAborted = 0;
			System.Console.WriteLine( " ");
			m_bOutputLines = false;
		}

		public bool outputLines()
		{
			return( m_bOutputLines);
		}

		public RCODE reportProgress(
			ref RestoreAction 	peRestoreAction,
			ulong						ulBytesToDo,
			ulong						ulBytesDone)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;

			System.Console.Write( "Bytes To Restore: {0}, Bytes Restored: {1}, TRCmit: {2}, TRAbrt: {3}\r",
				ulBytesToDo, ulBytesDone, m_ulNumTransCommitted, m_ulNumTransAborted);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportError(
			ref RestoreAction 	peRestoreAction,
			RCODE						rcErr)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;

			System.Console.WriteLine( "\nError reported: {0}", rcErr);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportBeginTrans(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;

			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportCommitTrans(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			m_ulNumTransCommitted++;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportAbortTrans(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			m_ulNumTransAborted++;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportBlockChainFree(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			ulong						ulMaintDocNum,
			uint						uiStartBlkAddr,
			uint						uiEndBlkAddr,
			uint						uiCount)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportIndexSuspend(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiIndexNum)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportIndexResume(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiIndexNum)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportReduce(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCount)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportUpgrade(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiOldDbVersion,
			uint						uiNewDbVersion)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportOpenRflFile(
			ref RestoreAction 	peRestoreAction,
			uint						uiFileNum)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportRflRead(
			ref RestoreAction 	peRestoreAction,
			uint						uiFileNum,
			uint						uiBytesRead)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportEnableEncryption(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportWrapKey(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportSetNextNodeId(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNextNodeId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeSetMetaValue(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			ulong						ulMetaValue)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeSetPrefixId(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			uint						uiAttrNameId,
			uint						uiPrefixId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeFlagsUpdate(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			uint						uiFlags,
			bool						bAdd)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportAttributeSetValue(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulElementNodeId,
			uint						uiAttrNameId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeSetValue(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeUpdate(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportInsertBefore(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulParentId,
			ulong						ulNewChildId,
			ulong						ulRefChildId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeCreate(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulRefNodeId,
			eDomNodeType			eNodeType,
			uint						uiNameId,
			eNodeInsertLoc			eLocation)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeChildrenDelete(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			uint						uiNameId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportAttributeDelete(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulElementNodeId,
			uint						uiAttrNameId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportNodeDelete(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportDocumentDone(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulDocumentId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportRollOverDbKey(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId)
		{
			peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
			return( RCODE.NE_XFLM_OK);
		}
	}
}
