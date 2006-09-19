//------------------------------------------------------------------------------
// Desc:	Db System
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
	/// Valid database versions
	/// </summary>
	public enum DBVersions
	{
		/// <summary>Version 5.12</summary>
		XFLM_VER_5_12					= 512,
		/// <summary>Current database version number</summary>
		XFLM_CURRENT_VERSION_NUM	= XFLM_VER_5_12
	}

	/// <summary>
	/// Valid languages
	/// </summary>
	public enum Languages
	{
		/// <summary>English, United States</summary>
		FLM_US_LANG 			= 0,
		/// <summary>Afrikaans</summary>
		FLM_AF_LANG 			= 1,
		/// <summary>Arabic</summary>
		FLM_AR_LANG 			= 2,
		/// <summary>Catalan</summary>
		FLM_CA_LANG 			= 3,
		/// <summary>Croatian</summary>
		FLM_HR_LANG 			= 4,
		/// <summary>Czech</summary>
		FLM_CZ_LANG 			= 5,
		/// <summary>Danish</summary>
		FLM_DK_LANG 			= 6,
		/// <summary>Dutch</summary>
		FLM_NL_LANG 			= 7,
		/// <summary>English, Australia</summary>
		FLM_OZ_LANG 			= 8,
		/// <summary>English, Canada</summary>
		FLM_CE_LANG 			= 9,
		/// <summary>English, United Kingdom</summary>
		FLM_UK_LANG 			= 10,
		/// <summary>Farsi</summary>
		FLM_FA_LANG 			= 11,
		/// <summary>Finnish</summary>
		FLM_SU_LANG 			= 12,
		/// <summary>French, Canada</summary>
		FLM_CF_LANG 			= 13,
		/// <summary>French, France</summary>
		FLM_FR_LANG 			= 14,
		/// <summary>Galician</summary>
		FLM_GA_LANG 			= 15,
		/// <summary>German, Germany</summary>
		FLM_DE_LANG 			= 16,
		/// <summary>German, Switzerland</summary>
		FLM_SD_LANG 			= 17,
		/// <summary>Greek</summary>
		FLM_GR_LANG 			= 18,
		/// <summary>Hebrew</summary>
		FLM_HE_LANG 			= 19,
		/// <summary>Hungarian</summary>
		FLM_HU_LANG 			= 20,
		/// <summary>Icelandic</summary>
		FLM_IS_LANG 			= 21,
		/// <summary>Italian</summary>
		FLM_IT_LANG 			= 22,
		/// <summary>Norwegian</summary>
		FLM_NO_LANG 			= 23,
		/// <summary>Polish</summary>
		FLM_PL_LANG 			= 24,
		/// <summary>Portuguese, Brazil</summary>
		FLM_BR_LANG 			= 25,
		/// <summary>Portuguese, Portugal</summary>
		FLM_PO_LANG 			= 26,
		/// <summary>Russian</summary>
		FLM_RU_LANG 			= 27,
		/// <summary>Slovak</summary>
		FLM_SL_LANG 			= 28,
		/// <summary>Spanish</summary>
		FLM_ES_LANG 			= 29,
		/// <summary>Swedish</summary>
		FLM_SV_LANG 			= 30,
		/// <summary>Ukrainian</summary>
		FLM_YK_LANG 			= 31,
		/// <summary>Urdu</summary>
		FLM_UR_LANG 			= 32,
		/// <summary>Turkey</summary>
		FLM_TK_LANG 			= 33,
		/// <summary>Japanese</summary>
		FLM_JP_LANG 			= 34,
		/// <summary>Korean</summary>
		FLM_KO_LANG 			= 35,
		/// <summary>Chinese-Traditional</summary>
		FLM_CT_LANG 			= 36,
		/// <summary>Chinese-Simplified</summary>
		FLM_CS_LANG 			= 37,
		/// <summary>another Asian language</summary>
		FLM_LA_LANG 			= 38
	}

	/// <summary>
	/// Types of logical files.  These are defined in xflaim.h.  If they
	/// are changed in xflaim.h, they need to be changed here as well.
	/// </summary>
	public enum eLFileType
	{
		/// <summary>Invalid type</summary>
		XFLM_LF_INVALID = 0,
		/// <summary>Collection</summary>
		XFLM_LF_COLLECTION,
		/// <summary>Index</summary>
		XFLM_LF_INDEX
	}

	/// <summary>
	/// Create options for creating a database
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public class CREATE_OPTS
	{

		/// <summary>
		/// Block size, may be 4096 or 8192.
		/// </summary>
		public uint 		uiBlockSize;

		/// <summary>
		/// Database version number.  Should be one of <see cref="DBVersions"/>.
		/// </summary>
		public uint 		uiVersionNum;

		/// <summary>
		/// Minimum Roll-forward log file size.
		/// </summary>
		public uint 		uiMinRflFileSize;

		/// <summary>
		/// Maximum Roll-forward log file size.
		/// </summary>
		public uint 		uiMaxRflFileSize;

		/// <summary>
		/// Flag indicating whether roll-forward log files should
		/// be kept or reused.
		/// </summary>
		public int 			bKeepRflFiles;

		/// <summary>
		/// Flag indicating whether aborted transactions should be
		/// logged to the roll-forward log.
		/// </summary>
		public int 			bLogAbortedTransToRfl;

		/// <summary>
		/// Default language for the database.  Should be one of <see cref="Languages"/>
		/// </summary>
		public uint 		uiDefaultLanguage;
	}

	/// <remarks>
	/// XFLAIM Exception class.
	/// </remarks>
	public class XFlaimException : Exception 
	{
		/// <summary>
		/// XFLAIM Exception that returns an RCODE.
		/// </summary>
		/// <param name="iRcode">The error code that occurred.</param>
		public XFlaimException( 
			int	iRcode)
		{
			m_rc = (RCODE)iRcode;
			m_message = null;
		}

		/// <summary>
		/// XFLAIM Exception that returns a message.
		/// </summary>
		/// <param name="message">Message explaining cause of exception.</param>
		public XFlaimException(
			string 		message)
		{
			m_message = message;
			m_rc = 0;
		}

		
		/// <summary>
		/// Returns the error code that caused the exception to be thrown.
		/// </summary>
		/// <returns>
		/// The error code that caused the exception.  If zero is returned
		/// there is no message associated with this exception.  Instead,
		/// the application should call <see cref="getString"/> to get
		/// the message that explains the cause of the exception.
		/// </returns>
		public RCODE getRCode()
		{
			return m_rc;
		}

		/// <summary>
		/// Returns the string that explains the cause of the exception.
		/// </summary>
		/// <returns>
		/// The string that explains the cause of the exception.  If null
		/// is returned, there is no message associated with this exception.
		/// Instead, the application should call <see cref="getRCode"/> to
		/// get the error code that caused the exception to be thrown.
		/// </returns>
		public string getString()
		{
			return m_message;
		}
	
		private string		m_message;
		private RCODE	m_rc;
	}

	/// <remarks>
	/// The DbSystem class provides a number of methods that allow C#
	/// applications to access the XFlaim development environment.
	/// </remarks>
	public class DbSystem
	{

		/// <summary>
		/// DbSystem constructor.
		/// </summary>
		public DbSystem()
		{
			int	rc = 0;

			if (( rc = xflaim_DbSystem_createDbSystem( out m_pDbSystem)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// DbSystem destructor.
		/// </summary>
		~DbSystem()
		{
			xflaim_DbSystem_Release( m_pDbSystem);
			m_pDbSystem = 0;
		}
		
		/// <summary>
		/// Called by <see cref="Db"/> class to silence compiler warning.
		/// Has no other important use!
		/// </summary>
		public ulong getDbSystem()
		{
			return m_pDbSystem;
		}

		/// <summary>
		/// Creates a new XFlaim database.
		/// </summary>
		/// <param name="sDbFileName">
		/// This is the name of the control file for the database.
		/// The control file is the primary database name.  It may include a full
		/// or partial directory name, or no directory name.  If a partial directory
		/// name or is included, it is assumed to be relative to the current working
		/// directory.  If no directory is specified, the file will be created in
		/// the current working directory.
		/// </param>
		/// <param name="sDataDir">
		/// The directory where the database data files are stored.
		/// If null, the data files will be stored in the same directory as the control
		/// file.
		/// </param>
		/// <param name="sRflDir">
		/// The directory where the roll forward log files should be
		/// stored.  If null, this defaults to the same directory where the control file
		/// exists.  Within this directory, XFLAIM expects a subdirectory to exist that
		/// holds the RFL files.  The subdirectory name is derived from the control
		/// file's base name. If the control file's base name has an extension
		/// of ".db", the ".db" is replaced with ".rfl".  If the control file's base
		/// name does not have an extension of ".db", an extension of ".rfl" is simply
		/// appended to the control file's base name.  For example, if the control file's
		/// base name is "MyDatabase.db", the subdirectory will be named "MyDatabase.rfl".
		/// If the control file's base name is "MyDatabase.xyz", the subdirectory will be
		/// named "MyDatabase.xyz.rfl".
		/// </param>
		/// <param name="sDictFileName">
		/// The name of a file which contains dictionary
		/// definition items.  May be null.  Ignored if sDictBuf is non-null.
		/// </param>
		/// <param name="sDictBuf">
		/// Contains dictionary definitions.  If null,
		/// sDictFileName is used.  If both sDictFileName and sDictBuf are null,
		/// the database is created with an empty dictionary.
		/// </param>
		/// <param name="createOpts">
		/// A structure that contains several parameters that affect the creation
		/// of the database.
		/// </param>
		/// <returns>An instance of a <see cref="Db"/> object.</returns>
		public Db dbCreate(
			string 				sDbFileName,
			string 				sDataDir,
			string 				sRflDir,
			string 				sDictFileName,
			string 				sDictBuf,
			CREATE_OPTS			createOpts)
		{
			Db			db = null;
			ulong 	pDb;
			int		rc;
		
			if ((rc = xflaim_DbSystem_dbCreate( m_pDbSystem, sDbFileName, sDataDir, sRflDir,
				sDictFileName, sDictBuf, createOpts, out pDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
		
			if (pDb != 0)
			{
				db = new Db( pDb, this);	
			}
			return( db);
		}
	
		/// <summary>
		/// Creates a new XFlaim database.
		/// </summary>
		/// <param name="sDbFileName">
		/// See documentation on <see cref="dbCreate"/>.
		/// </param>
		/// <param name="sDataDir">
		/// See documentation on <see cref="dbCreate"/>.
		/// </param>
		/// <param name="sRflDir">
		/// See documentation on <see cref="dbCreate"/>.
		/// </param>
		/// <param name="sPassword">
		/// Password for opening the database.  This is only needed
		/// if the database key is currently wrapped in a password instead of the
		/// local NICI storage key.
		/// </param>
		/// <param name="bAllowLimited">
		/// If true, allow access to a database whose database key cannot
		/// be unwrapped because the NICI storage key is not present.
		/// </param>
		/// <returns>An instance of a <see cref="Db"/> object.</returns>
		public Db dbOpen(
			string	sDbFileName,
			string	sDataDir,
			string	sRflDir,
			string	sPassword,
			bool		bAllowLimited)
		{
			Db			db = null;
			ulong 	pDb;
			int		rc;
		
			if ((rc = xflaim_DbSystem_dbOpen( m_pDbSystem, sDbFileName, sDataDir, sRflDir,
				sPassword, (int)(bAllowLimited ? 1 : 0), out pDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
		
			if (pDb != 0)
			{
				db = new Db( pDb, this);	
			}
			return( db);
		}
	
		/// <summary>
		/// Removes (deletes) an XFlaim database.
		/// </summary>
		/// <param name="sDbFileName">
		/// The name of the control file of the database to delete.
		/// For more information see <see cref="dbCreate"/>
		/// </param>
		/// <param name="sDataDir">
		/// The data file directory.  For more information see <see cref="dbCreate"/>.
		/// </param>
		/// <param name="sRflDir">
		/// The roll-forward log file directory.  For more information see <see cref="dbCreate"/>.
		/// </param>
		/// <param name="bRemoveRflFiles">
		/// If true, the roll forward log files will be deleted.
		/// </param>
		public void dbRemove(
			string	sDbFileName,
			string	sDataDir,
			string	sRflDir,
			bool		bRemoveRflFiles)
		{
			int	rc;

			if ((rc = xflaim_DbSystem_dbRemove( m_pDbSystem, sDbFileName, sDataDir, sRflDir,
				(int)(bRemoveRflFiles ? 1 : 0))) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Restores a previously backed up database. The <paramref name="sBackupPath"/> parameter
		/// and the <paramref name="restoreClient"/> parameter are mutually exclusive.  If the
		/// <paramref name="restoreClient"/> parameter is null, then the backup data will be read from
		/// <paramref name="sBackupPath"/>.  If <paramref name="restoreClient"/> is non-null,
		///  <paramref name="sBackupPath"/> is ignored.
		/// </summary>
		/// <param name="sDbPath">
		/// The name of the control file of the database to restore.
		/// </param>
		/// <param name="sDataDir">
		/// The data file directory.  For more information see <see cref="dbCreate"/>.
		/// </param>
		/// <param name="sRflDir">
		/// The roll-forward log file directory.  For more information see <see cref="dbCreate"/>.
		/// </param>
		/// <param name="sBackupPath">
		/// The path to the backup files.  This may be null.  If
		/// non-null, it specifies the directory where the backup files which are
		/// to be restored are found.  If null, the <paramref name="restoreClient"/> parameter must be
		/// non-null.
		/// </param>
		/// <param name="sPassword">
		/// Password for the backup.  If non-null, the database key in
		/// the backup was wrapped in a password instead of the local NICI storage
		/// key.  This allows the database to be restored to a different machine if
		/// desired.  If null, the database can only be restored to the same machine
		/// where it originally existed.
		/// </param>
		/// <param name="restoreClient">
		/// An object implementing the <see cref="RestoreClient"/> interface.  This may be null.  If
		/// non-null, it is an object that knows how to get the backup data.
		/// </param>
		/// <param name="restoreStatus">
		/// An object implementing <see cref="RestoreStatus"/> interface.  This may be null.  If
		/// non-null, it is a callback object whose methods will be called to report
		/// restore progress.
		/// </param>
		public void dbRestore(
			string			sDbPath,
			string			sDataDir,
			string			sRflDir,
			string			sBackupPath,
			string			sPassword,
			RestoreClient	restoreClient,
			RestoreStatus	restoreStatus)
		{
			int							rc;
			RestoreClientDelegate	restoreClientDelegate = null;
			RestoreClientCallback	fnRestoreClient = null;
			RestoreStatusDelegate	restoreStatusDelegate = null;
			RestoreStatusCallback	fnRestoreStatus = null;
			
			if (restoreClient != null)
			{
				restoreClientDelegate = new RestoreClientDelegate( restoreClient);
				fnRestoreClient = new RestoreClientCallback( restoreClientDelegate.funcRestoreClient);
			}
			if (restoreStatus != null)
			{
				restoreStatusDelegate = new RestoreStatusDelegate( restoreStatus);
				fnRestoreStatus = new RestoreStatusCallback( restoreStatusDelegate.funcRestoreStatus);
			}
		
			if ((rc = xflaim_DbSystem_dbRestore( m_pDbSystem, sDbPath, sDataDir, sRflDir, sBackupPath,
				sPassword, fnRestoreClient, fnRestoreStatus)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cpp
		private enum RestoreClientAction
		{
			RESTORE_OPEN_BACKUP_SET		= 1,
			RESTORE_OPEN_RFL_FILE		= 2,
			RESTORE_OPEN_INC_FILE		= 3,
			RESTORE_READ					= 4,
			RESTORE_CLOSE					= 5,
			RESTORE_ABORT_FILE			= 6
		}

		private delegate RCODE RestoreClientCallback(
			RestoreClientAction	eAction,
			uint						uiFileNum,
			uint						uiBytesRequested,
			IntPtr					pvBuffer,
			ref uint					puiBytesRead);
			
		private class RestoreClientDelegate
		{
			public RestoreClientDelegate(
				RestoreClient	restoreClient)
			{
				m_restoreClient = restoreClient; 
			}
			
			~RestoreClientDelegate()
			{
			}
			
			public RCODE funcRestoreClient(
				RestoreClientAction	eAction,
				uint						uiFileNum,
				uint						uiBytesRequested,
				IntPtr					pvBuffer,
				ref uint					uiBytesRead)
			{
				switch (eAction)
				{
					case RestoreClientAction.RESTORE_OPEN_BACKUP_SET:
						return( m_restoreClient.openBackupSet());
					case RestoreClientAction.RESTORE_OPEN_RFL_FILE:
						return( m_restoreClient.openRflFile( uiFileNum));
					case RestoreClientAction.RESTORE_OPEN_INC_FILE:
						return( m_restoreClient.openIncFile( uiFileNum));
					case RestoreClientAction.RESTORE_READ:
						return( m_restoreClient.read( uiBytesRequested, pvBuffer, ref uiBytesRead));
					case RestoreClientAction.RESTORE_CLOSE:
						return( m_restoreClient.close());
					case RestoreClientAction.RESTORE_ABORT_FILE:
						return( m_restoreClient.abortFile());
				}
				return( RCODE.NE_XFLM_INVALID_PARM);
			}
			
			private RestoreClient	m_restoreClient;
		}

		// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cpp
		private enum RestoreStatusAction
		{
			REPORT_PROGRESS					= 1,
			REPORT_ERROR						= 2,
			REPORT_BEGIN_TRANS				= 3,
			REPORT_COMMIT_TRANS				= 4,
			REPORT_ABORT_TRANS				= 5,
			REPORT_BLOCK_CHAIN_FREE			= 6,
			REPORT_INDEX_SUSPEND				= 7,
			REPORT_INDEX_RESUME				= 8,
			REPORT_REDUCE						= 9,
			REPORT_UPGRADE						= 10,
			REPORT_OPEN_RFL_FILE				= 11,
			REPORT_RFL_READ					= 12,
			REPORT_ENABLE_ENCRYPTION		= 13,
			REPORT_WRAP_KEY					= 14,
			REPORT_SET_NEXT_NODE_ID			= 15,
			REPORT_NODE_SET_META_VALUE		= 16,
			REPORT_NODE_SET_PREFIX_ID		= 17,
			REPORT_NODE_FLAGS_UPDATE		= 18,
			REPORT_ATTRIBUTE_SET_VALUE		= 19,
			REPORT_NODE_SET_VALUE			= 20,
			REPORT_NODE_UPDATE				= 21,
			REPORT_INSERT_BEFORE				= 22,
			REPORT_NODE_CREATE				= 23,
			REPORT_NODE_CHILDREN_DELETE	= 24,
			REPORT_ATTRIBUTE_DELETE			= 25,
			REPORT_NODE_DELETE				= 26,
			REPORT_DOCUMENT_DONE				= 27,
			REPORT_ROLL_OVER_DB_KEY			= 28
		}
		
		private delegate RCODE RestoreStatusCallback(
			RestoreStatusAction	eAction,
			ref RestoreAction 	eRestoreAction,
			ulong						ulTransId,
			ulong						ulLongNum1,
			ulong						ulLongNum2,
			ulong						ulLongNum3,
			uint						uiShortNum1,
			uint						uiShortNum2,
			uint						uiShortNum3,
			uint						uiShortNum4);

		private class RestoreStatusDelegate
		{
			public RestoreStatusDelegate(
				RestoreStatus	restoreStatus)
			{
				m_restoreStatus = restoreStatus; 
			}
			
			~RestoreStatusDelegate()
			{
			}
			
			public RCODE funcRestoreStatus(
				RestoreStatusAction	eAction,
				ref RestoreAction 	eRestoreAction,
				ulong						ulTransId,
				ulong						ulLongNum1,
				ulong						ulLongNum2,
				ulong						ulLongNum3,
				uint						uiShortNum1,
				uint						uiShortNum2,
				uint						uiShortNum3,
				uint						uiShortNum4)
			{
				switch (eAction)
				{
					case RestoreStatusAction.REPORT_PROGRESS:
						return( m_restoreStatus.reportProgress( ref eRestoreAction,
							ulLongNum1, ulLongNum2));
					case RestoreStatusAction.REPORT_ERROR:
						return( m_restoreStatus.reportError( ref eRestoreAction,
							(RCODE)uiShortNum1));
					case RestoreStatusAction.REPORT_BEGIN_TRANS:
						return( m_restoreStatus.reportBeginTrans( ref eRestoreAction,
							ulTransId));
					case RestoreStatusAction.REPORT_COMMIT_TRANS:
						return( m_restoreStatus.reportCommitTrans( ref eRestoreAction,
							ulTransId));
					case RestoreStatusAction.REPORT_ABORT_TRANS:
						return( m_restoreStatus.reportAbortTrans( ref eRestoreAction,
							ulTransId));
					case RestoreStatusAction.REPORT_BLOCK_CHAIN_FREE:
						return( m_restoreStatus.reportBlockChainFree( ref eRestoreAction,
							ulTransId, ulLongNum1, uiShortNum1, uiShortNum2, uiShortNum3));
					case RestoreStatusAction.REPORT_INDEX_SUSPEND:
						return( m_restoreStatus.reportIndexSuspend( ref eRestoreAction,
							ulTransId, uiShortNum1));
					case RestoreStatusAction.REPORT_INDEX_RESUME:
						return( m_restoreStatus.reportIndexResume( ref eRestoreAction,
							ulTransId, uiShortNum1));
					case RestoreStatusAction.REPORT_REDUCE:
						return( m_restoreStatus.reportReduce( ref eRestoreAction,
							ulTransId, uiShortNum1));
					case RestoreStatusAction.REPORT_UPGRADE:
						return( m_restoreStatus.reportUpgrade( ref eRestoreAction,
							ulTransId, uiShortNum1, uiShortNum2));
					case RestoreStatusAction.REPORT_OPEN_RFL_FILE:
						return( m_restoreStatus.reportOpenRflFile( ref eRestoreAction,
							uiShortNum1));
					case RestoreStatusAction.REPORT_RFL_READ:
						return( m_restoreStatus.reportRflRead( ref eRestoreAction,
							uiShortNum1, uiShortNum2));
					case RestoreStatusAction.REPORT_ENABLE_ENCRYPTION:
						return( m_restoreStatus.reportEnableEncryption( ref eRestoreAction,
							ulTransId));
					case RestoreStatusAction.REPORT_WRAP_KEY:
						return( m_restoreStatus.reportWrapKey( ref eRestoreAction,
							ulTransId));
					case RestoreStatusAction.REPORT_SET_NEXT_NODE_ID:
						return( m_restoreStatus.reportSetNextNodeId( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1));
					case RestoreStatusAction.REPORT_NODE_SET_META_VALUE:
						return( m_restoreStatus.reportNodeSetMetaValue( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, ulLongNum2));
					case RestoreStatusAction.REPORT_NODE_SET_PREFIX_ID:
						return( m_restoreStatus.reportNodeSetPrefixId( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, uiShortNum2, uiShortNum3));
					case RestoreStatusAction.REPORT_NODE_FLAGS_UPDATE:
						return( m_restoreStatus.reportNodeFlagsUpdate( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, uiShortNum2,
							(bool)(uiShortNum3 != 0 ? true : false)));
					case RestoreStatusAction.REPORT_ATTRIBUTE_SET_VALUE:
						return( m_restoreStatus.reportAttributeSetValue( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, uiShortNum2));
					case RestoreStatusAction.REPORT_NODE_SET_VALUE:
						return( m_restoreStatus.reportNodeSetValue( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1));
					case RestoreStatusAction.REPORT_NODE_UPDATE:
						return( m_restoreStatus.reportNodeUpdate( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1));
					case RestoreStatusAction.REPORT_INSERT_BEFORE:
						return( m_restoreStatus.reportInsertBefore( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, ulLongNum2, ulLongNum3));
					case RestoreStatusAction.REPORT_NODE_CREATE:
						return( m_restoreStatus.reportNodeCreate( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1,
							(eDomNodeType)uiShortNum2, uiShortNum3, (eNodeInsertLoc)uiShortNum4));
					case RestoreStatusAction.REPORT_NODE_CHILDREN_DELETE:
						return( m_restoreStatus.reportNodeChildrenDelete( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, uiShortNum2));
					case RestoreStatusAction.REPORT_ATTRIBUTE_DELETE:
						return( m_restoreStatus.reportAttributeDelete( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1, uiShortNum2));
					case RestoreStatusAction.REPORT_NODE_DELETE:
						return( m_restoreStatus.reportNodeDelete( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1));
					case RestoreStatusAction.REPORT_DOCUMENT_DONE:
						return( m_restoreStatus.reportDocumentDone( ref eRestoreAction,
							ulTransId, uiShortNum1, ulLongNum1));
					case RestoreStatusAction.REPORT_ROLL_OVER_DB_KEY:
						return( m_restoreStatus.reportRollOverDbKey( ref eRestoreAction,
							ulTransId));
				}
				return( RCODE.NE_XFLM_INVALID_PARM);
			}
			
			private RestoreStatus	m_restoreStatus;
		}

		/// <summary>
		/// Check for physical and logical corruptions on the specified database.
		/// </summary>
		/// <param name="sDbFileName">
		/// The name of the control file of the database to be checked.
		/// </param>
		/// <param name="sDataDir">
		/// The data file directory.  See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sRflDir">
		/// The roll-forward log file directory.  See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sPassword">
		/// Password for opening the database.  This is only needed
		/// if the database key is currently wrapped in a password instead of the
		/// local NICI storage key.
		/// </param>
		/// <param name="eFlags">
		/// Flags that control exactly what the operation checks.
		/// Should be a logical OR of the members of <see cref="DbCheckFlags"/>
		/// </param>
		/// <param name="checkStatus">
		/// An object that implements the <see cref="DbCheckStatus"/> interface.  Methods on
		/// this object will be called to report check progress and corruptions that are found.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DbInfo"/> object that contains various statistics that were
		/// gathered during the database check.
		/// </returns>
		public DbInfo dbCheck(
			string			sDbFileName,
			string			sDataDir,
			string			sRflDir,
			string			sPassword,
			DbCheckFlags	eFlags,
			DbCheckStatus	checkStatus)
		{
			int							rc;
			DbCheckStatusDelegate	dbCheckStatus = null;
			DbCheckStatusCallback	fnDbCheckStatus = null;
			ulong							pDbInfo;

			if (checkStatus != null)
			{
				dbCheckStatus = new DbCheckStatusDelegate( checkStatus);
				fnDbCheckStatus = new DbCheckStatusCallback( dbCheckStatus.funcDbCheckStatus);
			}

			if ((rc = xflaim_DbSystem_dbCheck( m_pDbSystem, sDbFileName, sDataDir,
				sRflDir, sPassword, eFlags, fnDbCheckStatus, out pDbInfo)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new DbInfo( pDbInfo));
		}

		private delegate RCODE DbCheckStatusCallback(
			int				bHaveProgressInfo,
			IntPtr			pProgressInfo,
			IntPtr			pCorruptInfo);

		private class DbCheckStatusDelegate
		{
			public DbCheckStatusDelegate(
				DbCheckStatus	dbCheckStatus)
			{
				m_dbCheckStatus = dbCheckStatus; 
			}
			
			~DbCheckStatusDelegate()
			{
			}
			
			public RCODE funcDbCheckStatus(
				int				bHaveProgressInfo,
				IntPtr			pProgressInfo,
				IntPtr			pCorruptInfo)
			{
				RCODE	rc = RCODE.NE_XFLM_OK;
	
				if (bHaveProgressInfo != 0)
				{
					rc = m_dbCheckStatus.reportProgress(
						(XFLM_PROGRESS_CHECK_INFO)Marshal.PtrToStructure( pProgressInfo,
																		typeof( XFLM_PROGRESS_CHECK_INFO)));
				}
				else
				{
					XFLM_CORRUPT_INFO	corruptInfo = new XFLM_CORRUPT_INFO();
					rc = m_dbCheckStatus.reportCheckErr(
						(XFLM_CORRUPT_INFO)Marshal.PtrToStructure( pCorruptInfo,
																	typeof( XFLM_CORRUPT_INFO)));
				}
				return( rc);
			}
			
			private DbCheckStatus	m_dbCheckStatus;
		}

		/// <summary>
		/// Makes a copy of an existing database.
		/// </summary>
		/// <param name="sSrcDbName">
		/// The name of the control file of the database to be copied.
		/// </param>
		/// <param name="sSrcDataDir">
		/// The directory where the data files for the source
		/// database are stored. See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sSrcRflDir">
		/// The directory where the RFL files for the source
		/// database are stored.  See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sDestDbName">
		/// The name of the control file that is to be created for the destination database.
		/// </param>
		/// <param name="sDestDataDir">
		/// The directory where the data files for the destination database will be stored.
		/// See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sDestRflDir">
		/// The directory where the RFL files for the destination database will be stored.
		/// See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="copyStatus">
		/// If non-null this is an object that implements the <see cref="DbCopyStatus"/>
		/// interface.  It is a callback object that is used to report copy progress.
		/// </param>
		public void dbCopy(
			string			sSrcDbName,
			string			sSrcDataDir,
			string			sSrcRflDir,
			string			sDestDbName,
			string			sDestDataDir,
			string			sDestRflDir,
			DbCopyStatus	copyStatus)
		{
			int						rc;
			DbCopyStatusDelegate	dbCopyStatus = null;
			DbCopyStatusCallback	fnDbCopyStatus = null;

			if (copyStatus != null)
			{
				dbCopyStatus = new DbCopyStatusDelegate( copyStatus);
				fnDbCopyStatus = new DbCopyStatusCallback( dbCopyStatus.funcDbCopyStatus);
			}
			if ((rc = xflaim_DbSystem_dbCopy( m_pDbSystem, sSrcDbName, sSrcDataDir, sSrcRflDir,
				sDestDbName, sDestDataDir, sDestRflDir, fnDbCopyStatus)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		private delegate RCODE DbCopyStatusCallback(
			ulong				ulBytesToCopy,
			ulong				ulBytesCopied,
			int				bNewSrcFile,
			IntPtr			pszSrcFileName,
			IntPtr			pszDestFileName);

		private class DbCopyStatusDelegate
		{
			public DbCopyStatusDelegate(
				DbCopyStatus	dbCopyStatus)
			{
				m_dbCopyStatus = dbCopyStatus; 
			}
			
			~DbCopyStatusDelegate()
			{
			}
			
			public RCODE funcDbCopyStatus(
				ulong				ulBytesToCopy,
				ulong				ulBytesCopied,
				int				bNewSrcFile,
				IntPtr			pszSrcFileName,
				IntPtr			pszDestFileName)
			{
				RCODE		rc = RCODE.NE_XFLM_OK;
				string	sSrcFileName = null;
				string	sDestFileName = null;
	
				if (bNewSrcFile != 0)
				{
					sSrcFileName = Marshal.PtrToStringAnsi( pszSrcFileName);
					sDestFileName = Marshal.PtrToStringAnsi( pszDestFileName);
				}
				rc = m_dbCopyStatus.dbCopyStatus( ulBytesToCopy, ulBytesCopied,
					sSrcFileName, sDestFileName);
				return( rc);
			}
			
			private DbCopyStatus	m_dbCopyStatus;
		}

		/// <summary>
		/// Rename a database.
		/// </summary>
		/// <param name="sDbName">
		/// The name of the control file of the database to be renamed.
		/// </param>
		/// <param name="sDataDir">
		/// The data file directory. See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sRflDir">
		/// The roll-forward log file directory.  See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sNewDbName">
		/// The new control file name for the database.
		/// </param>
		/// <param name="bOverwriteDestOk">
		/// If true, then if the data specified in sNewDbName already exists, it will be overwritten.
		/// </param>
		/// <param name="renameStatus">
		/// If non-null this is an object that implements the <see cref="DbRenameStatus"/>
		/// interface.  It is a callback object that is used to report rename progress.
		/// </param>
		public void dbRename(
			string				sDbName,
			string				sDataDir,
			string				sRflDir,
			string				sNewDbName,
			bool					bOverwriteDestOk,
			DbRenameStatus		renameStatus)
		{
			int							rc;
			DbRenameStatusDelegate	dbRenameStatus = null;
			DbRenameStatusCallback	fnDbRenameStatus = null;

			if (renameStatus != null)
			{
				dbRenameStatus = new DbRenameStatusDelegate( renameStatus);
				fnDbRenameStatus = new DbRenameStatusCallback( dbRenameStatus.funcDbRenameStatus);
			}

			if ((rc = xflaim_DbSystem_dbRename( m_pDbSystem, sDbName, sDataDir, sRflDir, sNewDbName,
				(int)(bOverwriteDestOk ? 1 : 0), fnDbRenameStatus)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		private delegate RCODE DbRenameStatusCallback(
			IntPtr			pszSrcFileName,
			IntPtr			pszDestFileName);

		private class DbRenameStatusDelegate
		{
			public DbRenameStatusDelegate(
				DbRenameStatus	dbRenameStatus)
			{
				m_dbRenameStatus = dbRenameStatus; 
			}
			
			~DbRenameStatusDelegate()
			{
			}
			
			public RCODE funcDbRenameStatus(
				IntPtr			pszSrcFileName,
				IntPtr			pszDestFileName)
			{
				return( m_dbRenameStatus.dbRenameStatus(
					Marshal.PtrToStringAnsi( pszSrcFileName),
					Marshal.PtrToStringAnsi( pszDestFileName)));
			}
			
			private DbRenameStatus	m_dbRenameStatus;
		}

		// PRIVATE METHODS THAT ARE IMPLEMENTED IN C AND C++

		[DllImport("xflaim")]
		private static extern int xflaim_DbSystem_createDbSystem(
			out ulong	ppDbSystem);

		[DllImport("xflaim")]
		private static extern int xflaim_DbSystem_Release(
			ulong	pDbSystem);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbCreate(
														ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string		pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszRflDir,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszDictFileName,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszDictBuf,
														CREATE_OPTS	pCreateOpts,
														out ulong	ppDb);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbOpen(
														ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string		pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszRflDir,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszPassword,
														int			bAllowLimited,
														out ulong	ppDb);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbRemove(
														ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string		pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 		pszRflDir,
														int			bRemoveRflFiles);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbRestore(
														ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string						pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszRflDir,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszBackupPath,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszPassword,
														RestoreClientCallback	fnRestoreClient,
														RestoreStatusCallback	fnRestoreStatus);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbCheck(
														ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string						pszDbName,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszRflDir,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszPassword,
														DbCheckFlags				eFlags,
														DbCheckStatusCallback	fnDbCheckStatus,
														out ulong					ppDbInfo);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbCopy(
														ulong						pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string					pszSrcDbName,
			[MarshalAs(UnmanagedType.LPStr)] string 					pszSrcDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 					pszSrcRflDir,
			[MarshalAs(UnmanagedType.LPStr)] string 					pszDestDbName,
			[MarshalAs(UnmanagedType.LPStr)] string 					pszDestDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 					pszDestRflDir,
														DbCopyStatusCallback	fnDbCopyStatus);

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern int xflaim_DbSystem_dbRename(
														ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr)] string						pszSrcDbName,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszSrcDataDir,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszSrcRflDir,
			[MarshalAs(UnmanagedType.LPStr)] string 						pszDestDbName,
														int							bOverwriteDestOk,
														DbRenameStatusCallback	fnDbRenameStatus);

		private ulong			m_pDbSystem;
	}
}
