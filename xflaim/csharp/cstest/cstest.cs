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
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace cstest
{
	public class Tester
	{
		private const string CREATE_DB_NAME = "create.db";
		private const string COPY_DB_NAME = "copy.db";
		private const string COPY2_DB_NAME = "copy2.db";
		private const string RENAME_DB_NAME = "rename.db";
		private const string RESTORE_DB_NAME = "restore.db";
		private const string BACKUP_PATH = "backup";
		private const string REBUILD_DB_NAME = "rebuild.db";
		private const string TEST_STREAM_STRING = "abcdefghijklmnopqrstuvwxyzABCDEFJHIJKLMNOPQRSTUVWXYZ0123456789";

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
			string	sSrcDbName,
			string	sDestDbName,
			DbSystem	dbSystem)
		{

			// Try copying the database

			MyDbCopyStatus	copyStatus = new MyDbCopyStatus();

			beginTest( "Copy Database Test (" + sSrcDbName + " --> " + sDestDbName + ")");
			try
			{
				dbSystem.dbCopy( sSrcDbName, null, null, sDestDbName, null, null, copyStatus);
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
				backup.backup( BACKUP_PATH, null, null, backupStatus);
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
		// Check database test.
		//--------------------------------------------------------------------------
		static bool checkDbTest(
			string	sDbName,
			DbSystem	dbSystem)
		{
			MyDbCheckStatus	dbCheckStatus = null;
			DbInfo				dbInfo = null;
			XFLM_DB_HDR			dbHdr = new XFLM_DB_HDR();

			// Try restoring the database

			beginTest( "Check Database Test (" + sDbName + ")");

			dbCheckStatus = new MyDbCheckStatus();
			try
			{
				dbInfo = dbSystem.dbCheck( sDbName, null, null, null,
					DbCheckFlags.XFLM_ONLINE | DbCheckFlags.XFLM_DO_LOGICAL_CHECK,
					dbCheckStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( dbCheckStatus.outputLines(), ex, "checking database");
				return( false);
			}

			dbInfo.getDbHdr( dbHdr);
			System.Console.Write( "\n");
			System.Console.WriteLine( "Signature............. {0}", dbHdr.szSignature);
			System.Console.WriteLine( "Database Version...... {0}", dbHdr.ui32DbVersion);
			System.Console.WriteLine( "Block Size............ {0}", dbHdr.ui16BlockSize);

			if (dbHdr.szSignature != "FLAIMDB")
			{
				endTest( true, false);
				System.Console.WriteLine( "Invalid signature in database header");
				return( false);
			}
			if (dbHdr.ui16BlockSize != 8192)
			{
				endTest( true, false);
				System.Console.WriteLine( "Invalid block size in database header");
				return( false);
			}
			if ((DBVersions)dbHdr.ui32DbVersion != DBVersions.XFLM_CURRENT_VERSION_NUM)
			{
				endTest( true, false);
				System.Console.WriteLine( "Invalid version in database header");
				return( false);
			}
			endTest( true, true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Rebuild database test.
		//--------------------------------------------------------------------------
		static bool rebuildDbTest(
			string	sSrcDbName,
			string	sDestDbName,
			DbSystem	dbSystem)
		{
			MyDbRebuildStatus	dbRebuildStatus = null;
			CREATE_OPTS			createOpts = null;

			// Try restoring the database

			beginTest( "Rebuild Database Test (" + sSrcDbName + " to " + sDestDbName + ")");

			dbRebuildStatus = new MyDbRebuildStatus();
			createOpts = new CREATE_OPTS();

			createOpts.uiBlockSize = 8192;
			createOpts.uiVersionNum = (uint)DBVersions.XFLM_CURRENT_VERSION_NUM;
			createOpts.uiMinRflFileSize = 2000000;
			createOpts.uiMaxRflFileSize = 20000000;
			createOpts.bKeepRflFiles = 1;
			createOpts.bLogAbortedTransToRfl = 1;
			createOpts.uiDefaultLanguage = (uint)Languages.FLM_DE_LANG;
			try
			{
				dbSystem.dbRebuild( sSrcDbName, null, sDestDbName, null, null,
					null, null, createOpts, dbRebuildStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( dbRebuildStatus.outputLines(), ex, "rebuilding database");
				return( false);
			}

			endTest( true, true);
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
		// Rename database test.
		//--------------------------------------------------------------------------
		static bool renameDbTest(
			string	sSrcDbName,
			string	sDestDbName,
			DbSystem	dbSystem)
		{

			// Try renaming the database

			MyDbRenameStatus	renameStatus = new MyDbRenameStatus();

			beginTest( "Rename Database Test (" + sSrcDbName + " --> " + sDestDbName + ")");
			try
			{
				dbSystem.dbRename( sSrcDbName, null, null, sDestDbName, true, renameStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( renameStatus.outputLines(), ex, "renaming database");
				return( false);
			}
			endTest( renameStatus.outputLines(), true);
			return( true);
		}

		//--------------------------------------------------------------------------
		// Stream tests
		//--------------------------------------------------------------------------
		static bool streamTests(
			DbSystem	dbSystem)
		{
			IStream			bufferStream;
			IStream			encoderStream;
			IStream			decoderStream;
			OStream			fileOStream;
			Stream			s;
			StreamReader	sr;
			string			sFileData;

			beginTest( "Creating IStream from buffer");
			try
			{
				bufferStream = dbSystem.openBufferIStream( TEST_STREAM_STRING);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBufferIStream");
				return( false);
			}
			endTest( false, true);

			beginTest( "Creating base 64 encoder stream");
			try
			{
				encoderStream = dbSystem.openBase64Encoder( bufferStream, true);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBase64Encoder");
				return( false);
			}
			endTest( false, true);

			beginTest( "Creating base 64 decoder stream");
			try
			{
				decoderStream = dbSystem.openBase64Decoder( encoderStream);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBase64Decoder");
				return( false);
			}
			endTest( false, true);

			beginTest( "Creating file output stream");
			try
			{
				fileOStream = dbSystem.openFileOStream( "Output_Stream", true);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openFileOStream");
				return( false);
			}
			endTest( false, true);

			beginTest( "Writing from input stream to output stream");
			try
			{
				dbSystem.writeToOStream( decoderStream, fileOStream);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling writeToOStream");
				return( false);
			}
			fileOStream.close();
			endTest( false, true);

			beginTest( "Comparing output stream data to original data");

			s = File.OpenRead( "Output_Stream");
			sr = new StreamReader( s);
			sFileData = sr.ReadLine();
			if (sFileData != TEST_STREAM_STRING)
			{
				endTest( false, false);
				System.Console.WriteLine( "Stream data does not match original string");
				System.Console.WriteLine( "File Data:\n[{0}]", sFileData);
				System.Console.WriteLine( "Original String:\n[{0}]", TEST_STREAM_STRING);
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

			if (!copyDbTest( CREATE_DB_NAME, COPY_DB_NAME, dbSystem))
			{
				return;
			}
			if (!copyDbTest( CREATE_DB_NAME, COPY2_DB_NAME, dbSystem))
			{
				return;
			}

			// Database rename test

			if (!renameDbTest( COPY2_DB_NAME, RENAME_DB_NAME, dbSystem))
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

			// Database rebuild test

			if (!rebuildDbTest( RESTORE_DB_NAME, REBUILD_DB_NAME, dbSystem))
			{
				return;
			}

			// Database check test

			if (!checkDbTest( CREATE_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDbTest( COPY_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDbTest( RESTORE_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDbTest( RENAME_DB_NAME, dbSystem))
			{
				return;
			}
			if (!checkDbTest( REBUILD_DB_NAME, dbSystem))
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
			if (!removeDbTest( dbSystem, RENAME_DB_NAME))
			{
				return;
			}
			if (!removeDbTest( dbSystem, REBUILD_DB_NAME))
			{
				return;
			}

			// Input and Output stream tests

			if (!streamTests( dbSystem))
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

	public class PrintCorruption
	{
		public static void printCorruption(
			XFLM_CORRUPT_INFO	corruptInfo)
		{
			System.Console.WriteLine( "\nCorruption Found: {0}, Locale: {1}",
				corruptInfo.eErrCode, corruptInfo.eErrLocale);
			if (corruptInfo.uiErrLfNumber != 0)
			{
				System.Console.WriteLine( "  Logical File Number...... {0} ({1})",
					corruptInfo.uiErrLfNumber, corruptInfo.eErrLfType);
				System.Console.WriteLine( "  B-Tree Level............. {0}",
					corruptInfo.uiErrBTreeLevel);
			}
			if (corruptInfo.uiErrBlkAddress != 0)
			{
				System.Console.WriteLine( "  Block Address............ {0:X})",
					corruptInfo.uiErrBlkAddress);
			}
			if (corruptInfo.uiErrParentBlkAddress != 0)
			{
				System.Console.WriteLine( "  Parent Block Address..... {0:X})",
					corruptInfo.uiErrParentBlkAddress);
			}
			if (corruptInfo.uiErrElmOffset != 0)
			{
				System.Console.WriteLine( "  Element Offset........... {0})",
					corruptInfo.uiErrElmOffset);
			}
			if (corruptInfo.ulErrNodeId != 0)
			{
				System.Console.WriteLine( "  Node ID.................. {0})",
					corruptInfo.ulErrNodeId);
			}
		}
	}

	public class MyDbCheckStatus : DbCheckStatus
	{
		public MyDbCheckStatus()
		{
			m_bOutputLines = false;
			System.Console.Write( "\n");
		}

		public RCODE reportProgress(
			XFLM_PROGRESS_CHECK_INFO	progressInfo)
		{
			if (progressInfo.bStartFlag != 0)
			{
				if (progressInfo.eCheckPhase == FlmCheckPhase.XFLM_CHECK_B_TREE)
				{
					System.Console.WriteLine( "\nChecking B-Tree: {0} ({1})",
						progressInfo.uiLfNumber, progressInfo.eLfType);
				}
				else
				{
					System.Console.WriteLine( "\nCheck Phase: {0}", progressInfo.eCheckPhase);
				}
			}
			System.Console.Write( "Bytes To Check: {0}, Bytes Checked: {1}\r",
				progressInfo.ulDatabaseSize, progressInfo.ulBytesExamined);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportCheckErr(
			XFLM_CORRUPT_INFO	corruptInfo)
		{
			PrintCorruption.printCorruption( corruptInfo);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public bool outputLines()
		{
			return( m_bOutputLines);
		}

		private bool	m_bOutputLines;
	}

	public class MyDbRebuildStatus : DbRebuildStatus
	{
		public MyDbRebuildStatus()
		{
			m_bOutputLines = false;
			System.Console.Write( "\n");
		}

		public RCODE reportRebuild(
			XFLM_REBUILD_INFO	rebuildInfo)
		{
			if (rebuildInfo.bStartFlag != 0)
			{
				System.Console.WriteLine( "\nRebuild Phase: {0}", rebuildInfo.eDoingFlag);
			}
			System.Console.Write( "Bytes To Do {0}, Bytes Done: {1}\r",
				rebuildInfo.ulDatabaseSize, rebuildInfo.ulBytesExamined);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public RCODE reportRebuildErr(
			XFLM_CORRUPT_INFO	corruptInfo)
		{
			PrintCorruption.printCorruption( corruptInfo);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public bool outputLines()
		{
			return( m_bOutputLines);
		}

		private bool	m_bOutputLines;
	}

	public class MyDbRenameStatus : DbRenameStatus
	{
		public MyDbRenameStatus()
		{
			m_bOutputLines = false;
			System.Console.Write( "\n");
		}

		public RCODE dbRenameStatus(
			string		sSrcFileName,
			string		sDestFileName)
		{
			System.Console.WriteLine( "Renaming {0} to {1}", sSrcFileName, sDestFileName);
			m_bOutputLines = true;
			return( RCODE.NE_XFLM_OK);
		}

		public bool outputLines()
		{
			return( m_bOutputLines);
		}

		private bool	m_bOutputLines;
	}
}
