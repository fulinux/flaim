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
	public enum DBVersions : uint
	{
		/// <summary>Version 5.12</summary>
		XFLM_VER_5_12					= 512,
		/// <summary>Current database version number</summary>
		XFLM_CURRENT_VERSION_NUM	= XFLM_VER_5_12
	}

	/// <summary>
	/// Valid languages
	/// </summary>
	public enum Languages : int
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
	public enum eLFileType : int
	{
		/// <summary>Invalid type</summary>
		XFLM_LF_INVALID = 0,
		/// <summary>Collection</summary>
		XFLM_LF_COLLECTION,
		/// <summary>Index</summary>
		XFLM_LF_INDEX
	}

	/// <summary>
	/// Create options for creating a database.
	/// IMPORTANT NOTE: This needs to be kept in sync with the same
	/// structure that is defined in xflaim.h
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public class XFLM_CREATE_OPTS
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
		/// <param name="rc">The error code that occurred.</param>
		public XFlaimException( 
			RCODE	rc)
		{
			m_rc = rc;
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
		private ulong	m_pDbSystem;	// Pointer to IF_DbSystem object in unmanaged space

		/// <summary>
		/// Maximum key size - keep in sync with definition in xflaim.h
		/// </summary>
		public const uint XFLM_MAX_KEY_SIZE = 1024;

		/// <summary>
		/// DbSystem constructor.
		/// </summary>
		public DbSystem()
		{
			RCODE	rc = 0;

			if (( rc = xflaim_DbSystem_createDbSystem( out m_pDbSystem)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystem_createDbSystem(
			out ulong	ppDbSystem);

		/// <summary>
		/// DbSystem destructor.
		/// </summary>
		~DbSystem()
		{
			xflaim_DbSystem_Release( m_pDbSystem);
			m_pDbSystem = 0;
		}
		
		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystem_Release(
			ulong	pDbSystem);

		/// <summary>
		/// Called by <see cref="Db"/> class to silence compiler warning.
		/// Has no other important use!
		/// </summary>
		internal ulong getDbSystem()
		{
			return m_pDbSystem;
		}

		//-----------------------------------------------------------------------------
		// dbCreate
		//-----------------------------------------------------------------------------

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
			XFLM_CREATE_OPTS	createOpts)
		{
			ulong 	pDb;
			RCODE		rc;
		
			if ((rc = xflaim_DbSystem_dbCreate( m_pDbSystem, sDbFileName, sDataDir, sRflDir,
				sDictFileName, sDictBuf, createOpts, out pDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new Db( pDb, this));
		}
	
		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbCreate(
			ulong					pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string				pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 				pszDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 				pszRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 				pszDictFileName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 				pszDictBuf,
			XFLM_CREATE_OPTS	pCreateOpts,
			out ulong			ppDb);

		//-----------------------------------------------------------------------------
		// dbOpen
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Opens an existing XFlaim database.
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
			ulong 	pDb;
			RCODE		rc;
		
			if ((rc = xflaim_DbSystem_dbOpen( m_pDbSystem, sDbFileName, sDataDir, sRflDir,
				sPassword, (int)(bAllowLimited ? 1 : 0), out pDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
		
			return( new Db( pDb, this));
		}
	
		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbOpen(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 		pszDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 		pszRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 		pszPassword,
			int			bAllowLimited,
			out ulong	ppDb);

		//-----------------------------------------------------------------------------
		// dbRemove
		//-----------------------------------------------------------------------------

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
			RCODE	rc;

			if ((rc = xflaim_DbSystem_dbRemove( m_pDbSystem, sDbFileName, sDataDir, sRflDir,
				(int)(bRemoveRflFiles ? 1 : 0))) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbRemove(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 		pszDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 		pszRflDir,
			int			bRemoveRflFiles);

		//-----------------------------------------------------------------------------
		// dbRestore
		//-----------------------------------------------------------------------------

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
			RCODE							rc;
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

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbRestore(
			ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string						pszDbFileName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszBackupPath,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszPassword,
			RestoreClientCallback	fnRestoreClient,
			RestoreStatusCallback	fnRestoreStatus);

		// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cpp
		private enum RestoreClientAction : int
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
		private enum RestoreStatusAction : int
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

		//-----------------------------------------------------------------------------
		// dbCheck
		//-----------------------------------------------------------------------------

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
			RCODE							rc;
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

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbCheck(
			ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string						pszDbName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszPassword,
			DbCheckFlags				eFlags,
			DbCheckStatusCallback	fnDbCheckStatus,
			out ulong					ppDbInfo);

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
					rc = m_dbCheckStatus.reportCheckErr(
						(XFLM_CORRUPT_INFO)Marshal.PtrToStructure( pCorruptInfo,
						typeof( XFLM_CORRUPT_INFO)));
				}
				return( rc);
			}
			
			private DbCheckStatus	m_dbCheckStatus;
		}

		//-----------------------------------------------------------------------------
		// dbCopy
		//-----------------------------------------------------------------------------

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
			RCODE						rc;
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

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbCopy(
			ulong						pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string					pszSrcDbName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 					pszSrcDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 					pszSrcRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 					pszDestDbName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 					pszDestDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 					pszDestRflDir,
			DbCopyStatusCallback	fnDbCopyStatus);

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

		//-----------------------------------------------------------------------------
		// dbRename
		//-----------------------------------------------------------------------------

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
			RCODE							rc;
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

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbRename(
			ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string						pszSrcDbName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszSrcDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszSrcRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDestDbName,
			int							bOverwriteDestOk,
			DbRenameStatusCallback	fnDbRenameStatus);

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

		//-----------------------------------------------------------------------------
		// dbRebuild
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Rebuild a database.
		/// </summary>
		/// <param name="sSourceDbPath">
		/// The name of the control file of the database that is to be rebuilt.
		/// </param>
		/// <param name="sSourceDataDir">
		/// The data file directory.  See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sDestDbPath">
		/// The name of the control file of the destination
		/// database that is to be built from the source database.
		/// </param>
		/// <param name="sDestDataDir">
		/// The destination database's data file directory.  See <see cref="dbCreate"/> for
		/// more information.
		/// </param>
		/// <param name="sDestRflDir">
		/// The destination database's roll-forward log
		/// directory.  See <see cref="dbCreate"/> for more information.
		/// </param>
		/// <param name="sDictPath">
		/// The name of a file containing dictionary definitions that
		/// are to be put into the destination database when it is created.
		/// May be null.
		/// </param>
		/// <param name="sPassword">
		/// Password for opening the source database.  This is only needed
		/// if the database key is currently wrapped in a password instead of the
		/// local NICI storage key.  May be null.
		/// </param>
		/// <param name="createOpts">
		/// A <see cref="XFLM_CREATE_OPTS"/> object that contains several parameters that
		/// are used in the creation of the destination database.
		/// </param>
		/// <param name="rebuildStatus">
		/// If non-null this is an object that implements the <see cref="DbRebuildStatus"/>
		/// interface.  It is a callback object that is used to report rebuild progress.
		/// </param>
		public void dbRebuild(
			string				sSourceDbPath,
			string				sSourceDataDir,
			string				sDestDbPath,
			string				sDestDataDir,
			string				sDestRflDir,
			string				sDictPath,
			string				sPassword,
			XFLM_CREATE_OPTS	createOpts,
			DbRebuildStatus	rebuildStatus)
		{
			RCODE							rc;
			DbRebuildStatusDelegate	dbRebuildStatus = null;
			DbRebuildStatusCallback	fnDbRebuildStatus = null;

			if (rebuildStatus != null)
			{
				dbRebuildStatus = new DbRebuildStatusDelegate( rebuildStatus);
				fnDbRebuildStatus = new DbRebuildStatusCallback( dbRebuildStatus.funcDbRebuildStatus);
			}

			if ((rc = xflaim_DbSystem_dbRebuild( m_pDbSystem, sSourceDbPath, sSourceDataDir, sDestDbPath,
				sDestDataDir, sDestRflDir, sDictPath, sPassword,
				createOpts, fnDbRebuildStatus)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_dbRebuild(
			ulong							pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string						pszSourceDbPath,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszSourceDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDestDbPath,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDestDataDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDestRflDir,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszDictPath,
			[MarshalAs(UnmanagedType.LPStr), In]
			string 						pszPassword,
			XFLM_CREATE_OPTS			pCreateOpts,
			DbRebuildStatusCallback	fnDbRebuildStatus);

		private delegate RCODE DbRebuildStatusCallback(
			int				bHaveRebuildInfo,
			IntPtr			pRebuildInfo,
			IntPtr			pCorruptInfo);

		private class DbRebuildStatusDelegate
		{
			public DbRebuildStatusDelegate(
				DbRebuildStatus	dbRebuildStatus)
			{
				m_dbRebuildStatus = dbRebuildStatus; 
			}
			
			~DbRebuildStatusDelegate()
			{
			}
			
			public RCODE funcDbRebuildStatus(
				int				bHaveRebuildInfo,
				IntPtr			pRebuildInfo,
				IntPtr			pCorruptInfo)
			{
				RCODE	rc = RCODE.NE_XFLM_OK;
	
				if (bHaveRebuildInfo != 0)
				{
					rc = m_dbRebuildStatus.reportRebuild(
						(XFLM_REBUILD_INFO)Marshal.PtrToStructure( pRebuildInfo,
						typeof( XFLM_REBUILD_INFO)));
				}
				else
				{
					rc = m_dbRebuildStatus.reportRebuildErr(
						(XFLM_CORRUPT_INFO)Marshal.PtrToStructure( pCorruptInfo,
						typeof( XFLM_CORRUPT_INFO)));
				}
				return( rc);
			}

			private DbRebuildStatus	m_dbRebuildStatus;
		}

		//-----------------------------------------------------------------------------
		// openBufferIStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that reads from a string buffer.
		/// </summary>
		/// <param name="sBuffer">
		/// String the input stream is to read from.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openBufferIStream(
			string	sBuffer)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openBufferIStream( sBuffer, out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openBufferIStream(
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszBuffer,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openFileIStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that reads from a file.
		/// </summary>
		/// <param name="sFileName">
		/// Name of the file the input stream is to read from.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openFileIStream(
			string	sFileName)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openFileIStream( m_pDbSystem, sFileName, out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openFileIStream(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszFileName,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openMultiFileIStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that reads from multiple files.
		/// </summary>
		/// <param name="sDirectory">
		/// Name of the directory where the multiple files are to be found.
		/// </param>
		/// <param name="sBaseName">
		/// Base name of the input files.  Files that constitute the
		/// input stream are sBaseName, sBaseName.00000001, sBaseName.00000002, etc. - where
		/// the extension is a Hex number.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openMultiFileIStream(
			string	sDirectory,
			string	sBaseName)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openMultiFileIStream( m_pDbSystem, sDirectory,
				sBaseName, out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openMultiFileIStream(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszDirectory,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszBaseName,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openBufferedIStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that buffers an existing input stream.
		/// </summary>
		/// <param name="inputIStream">
		/// Input stream that is to be buffered.
		/// </param>
		/// <param name="uiBufferSize">
		/// iBufferSize The size (in bytes) of the buffer to use for the
		/// input stream.  Data will be read into the buffer in chunks of this size.
		/// This will help performance by preventing lots of smaller reads from
		/// the original input stream.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openBufferedIStream(
			IStream		inputIStream,
			uint			uiBufferSize)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openBufferedIStream( m_pDbSystem,
				inputIStream.getIStream(), uiBufferSize, out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openBufferedIStream(
			ulong			pDbSystem,
			ulong			pInputIStream,
			uint			uiBufferSize,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openUncompressingIStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that decompresses data from another input stream.  It
		/// is assumed that data coming out of the other input stream is compressed.
		/// </summary>
		/// <param name="inputIStream">
		/// Input stream whose data is to be decompressed.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openUncompressingIStream(
			IStream	inputIStream)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openUncompressingIStream( m_pDbSystem,
				inputIStream.getIStream(), out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openUncompressingIStream(
			ulong			pDbSystem,
			ulong			pInputIStream,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openBase64Encoder
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that encodes data from another input stream into
		/// base 64 encoded binary.  Data read from the stream object returned by
		/// this method will be base 64 encoded.
		/// </summary>
		/// <param name="inputIStream">
		/// Input stream whose data is to be base 64 encoded.
		/// </param>
		/// <param name="bInsertLineBreaks">
		/// Flag indicating whether or not line breaks
		/// should be inserted into the data as it is base 64 encoded.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openBase64Encoder(
			IStream	inputIStream,
			bool		bInsertLineBreaks)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openBase64Encoder( m_pDbSystem,
				inputIStream.getIStream(), (int)(bInsertLineBreaks ? 1 : 0),
				out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openBase64Encoder(
			ulong			pDbSystem,
			ulong			pInputIStream,
			int			bInsertLineBreaks,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openBase64Decoder
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an input stream that decodes data from another input stream.  It is
		/// assumed that data read from the original input stream is base 64
		/// encoded.
		/// </summary>
		/// <param name="inputIStream">
		/// Input stream whose data is to be decoded.  It is assumed that data read
		/// from this stream is base 64 encoded.
		/// </param>
		/// <returns>
		/// Returns an <see cref="IStream"/> object that can then be passed to
		/// methods which require an IStream object.
		/// </returns>
		public IStream openBase64Decoder(
			IStream	inputIStream)
		{
			RCODE		rc;
			ulong		pIStream = 0;

			if ((rc = xflaim_DbSystem_openBase64Decoder( m_pDbSystem,
				inputIStream.getIStream(), out pIStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new IStream( pIStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openBase64Decoder(
			ulong			pDbSystem,
			ulong			pInputIStream,
			out ulong	ppIStream);

		//-----------------------------------------------------------------------------
		// openFileOStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open an output stream that writes data to a file.
		/// </summary>
		/// <param name="sFileName">
		/// Name of file to write data to.
		/// </param>
		/// <param name="bTruncateIfExists">
		/// Flag indicating whether or not the output file
		/// should be truncated if it already exists.  If false, the file will be
		/// appended to.
		/// </param>
		/// <returns>
		/// Returns an <see cref="OStream"/> object that can then be passed to
		/// methods which require an OStream object.
		/// </returns>
		public OStream openFileOStream(
			string	sFileName,
			bool		bTruncateIfExists)
		{
			RCODE		rc;
			ulong		pOStream = 0;

			if ((rc = xflaim_DbSystem_openFileOStream( m_pDbSystem,
				sFileName, (int)(bTruncateIfExists ? 1 : 0), out pOStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new OStream( pOStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openFileOStream(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszFileName,
			int			bTruncateIfExists,
			out ulong	ppOStream);

		//-----------------------------------------------------------------------------
		// openMultiFileOStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open a multi-file output stream.  Data is written to one or more files.
		/// </summary>
		/// <param name="sDirectory">
		/// Directory where output files are to be created.
		/// </param>
		/// <param name="sBaseName">
		/// Base name for creating file names.  The first file will
		/// be called sBaseName.  Subsequent files will be named sBaseName.00000001,
		/// sBaseName.00000002, etc.  The extension is a hex number.
		/// </param>
		/// <param name="uiMaxFileSize">
		/// Maximum number of bytes to write to each file in the multi-file set.
		/// </param>
		/// <param name="bOkToOverwrite">
		/// Flag indicating whether or not the output files
		/// should be overwritten if they already exist.
		/// </param>
		/// <returns>
		/// Returns an <see cref="OStream"/> object that can then be passed to
		/// methods which require an OStream object.
		/// </returns>
		public OStream openMultiFileOStream(
			string	sDirectory,
			string	sBaseName,
			uint		uiMaxFileSize,
			bool		bOkToOverwrite)
		{
			RCODE		rc;
			ulong		pOStream = 0;

			if ((rc = xflaim_DbSystem_openMultiFileOStream( m_pDbSystem,
				sDirectory, sBaseName, uiMaxFileSize,
				(int)(bOkToOverwrite ? 1 : 0), out pOStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new OStream( pOStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openMultiFileOStream(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszDirectory,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		sBaseName,
			uint			uiMaxFileSize,
			int			bOkToOverwrite,
			out ulong	ppOStream);

		//-----------------------------------------------------------------------------
		// removeMultiFileStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Remove a multi-file output stream from disk.
		/// </summary>
		/// <param name="sDirectory">
		/// Directory where the files belonging to the output stream are located.
		/// </param>
		/// <param name="sBaseName">
		/// Base name for files belonging to the output stream.  The first file will
		/// be called sBaseName.  Subsequent files will be named sBaseName.00000001,
		/// sBaseName.00000002, etc.  The extension is a hex number.
		/// </param>
		public void removeMultiFileStream(
			string	sDirectory,
			string	sBaseName)
		{
			RCODE	rc;

			if ((rc = xflaim_DbSystem_removeMultiFileStream(  m_pDbSystem,
				sDirectory, sBaseName)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_removeMultiFileStream(
			ulong			pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		pszDirectory,
			[MarshalAs(UnmanagedType.LPStr), In]
			string		sBaseName);

		//-----------------------------------------------------------------------------
		// openBufferedOStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open a buffered output stream.  A buffer is allocated for writing data to
		/// the original output stream.  Instead of writing small chunks of data to
		/// the original output stream, it is first gathered into the output buffer.
		/// When the output buffer fills, the entire buffer is sent to the original
		/// output stream with a single write.  The idea is that by buffering the
		/// output data, performance can be improved.
		/// </summary>
		/// <param name="inputOStream">
		/// Output stream that the data is ultimately going to be written to - but it
		/// will be buffered before being written.
		/// </param>
		/// <param name="uiBufferSize">
		/// Size of the buffer to be used for buffering output.
		/// </param>
		/// <returns>
		/// Returns an <see cref="OStream"/> object that can then be passed to
		/// methods which require an OStream object.
		/// </returns>
		public OStream openBufferedOStream(
			OStream	inputOStream,
			uint		uiBufferSize)
		{
			RCODE		rc;
			ulong		pOStream = 0;

			if ((rc = xflaim_DbSystem_openBufferedOStream( m_pDbSystem,
				inputOStream.getOStream(), uiBufferSize, out pOStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new OStream( pOStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openBufferedOStream(
			ulong			pDbSystem,
			ulong			pInputOStream,
			uint			uiBufferSize,
			out ulong	ppOStream);

		//-----------------------------------------------------------------------------
		// openCompressingOStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Open a compressing output stream.  Data is compressed before writing it
		/// out to the passed in output stream object.
		/// </summary>
		/// <param name="inputOStream">
		/// Output stream that the data is ultimately going to be written to - but it
		/// will be compressed before being written.
		/// </param>
		/// <returns>
		/// Returns an <see cref="OStream"/> object that can then be passed to
		/// methods which require an OStream object.
		/// </returns>
		public OStream openCompressingOStream(
			OStream	inputOStream)
		{
			RCODE		rc;
			ulong		pOStream = 0;

			if ((rc = xflaim_DbSystem_openCompressingOStream( m_pDbSystem,
				inputOStream.getOStream(), out pOStream)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new OStream( pOStream, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_openCompressingOStream(
			ulong			pDbSystem,
			ulong			pInputOStream,
			out ulong	ppOStream);

		//-----------------------------------------------------------------------------
		// writeToOStream
		//-----------------------------------------------------------------------------

		/// <summary>
		/// Read data from an input stream and write it out to an output stream.  This
		/// is a quick way to copy all data from an input stream to an output stream.
		/// </summary>
		/// <param name="istream">
		/// Input stream data is to be read from.
		/// </param>
		/// <param name="ostream">
		/// Output stream data is to be written to.
		/// </param>
		public void writeToOStream(
			IStream	istream,
			OStream	ostream)
		{
			RCODE	rc;
			
			if ((rc = xflaim_DbSystem_writeToOStream( m_pDbSystem,
				istream.getIStream(), ostream.getOStream())) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_writeToOStream(
			ulong			pDbSystem,
			ulong			pIStream,
			ulong			pOStream);

//-----------------------------------------------------------------------------
// createDataVector
//-----------------------------------------------------------------------------

		/// <summary>
		/// Create a <see cref="DataVector"/> object.
		/// </summary>
		/// <returns>Returns a <see cref="DataVector"/> objecdt</returns>
		public DataVector createDataVector()
		{
			RCODE	rc;
			ulong	pDataVector;
			
			if ((rc = xflaim_DbSystem_createDataVector( m_pDbSystem, out pDataVector)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new DataVector( pDataVector, this));
		}

		[DllImport("xflaim",CharSet=CharSet.Ansi)]
		private static extern RCODE xflaim_DbSystem_createDataVector(
			ulong			pDbSystem,
			out ulong	ppDataVector);

//-----------------------------------------------------------------------------
// freeUnmanagedMem
//-----------------------------------------------------------------------------

		internal void freeUnmanagedMem(
			IntPtr	pMem)
		{
			xflaim_DbSystem_freeUnmanagedMem( m_pDbSystem, pMem);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DbSystem_freeUnmanagedMem(
			ulong		pDbSystem,
			IntPtr	pMem);

//-----------------------------------------------------------------------------
// updateIniFile
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a parameter in the .INI file for XFLAIM.
		/// </summary>
		/// <param name="sParamName">
		/// Name of parameter whose value is to be set.
		/// </param>
		/// <param name="sValue">
		/// Value the parameter is to be set to.
		/// </param>
		public void updateIniFile(
			string	sParamName,
			string	sValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DbSystem_updateIniFile( m_pDbSystem, sParamName, sValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystem_updateIniFile(
			ulong		pDbSystem,
			[MarshalAs(UnmanagedType.LPStr), In]
			string	sParamName,
			[MarshalAs(UnmanagedType.LPStr), In]
			string	sValue);

//-----------------------------------------------------------------------------
// dbDup
//-----------------------------------------------------------------------------

		/// <summary>
		/// Duplicate a <see cref="Db"/> object.  This method is a quicker way to open
		/// a database than calling the <see cref="dbOpen"/> method.  If the application has
		/// already opened a database, it may pass the Db object it obtained
		/// into this method to get another Db object.
		/// </summary>
		/// <param name="dbToDup">
		/// Db object to duplicate.
		/// </param>
		/// <returns>Returns a new <see cref="Db"/> object.</returns>
		public Db dbDup(
			Db			dbToDup)
		{
			RCODE	rc;
			ulong	pDupDb;

			if ((rc = xflaim_DbSystem_dbDup( m_pDbSystem, dbToDup.getDb(), out pDupDb)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( new Db( pDupDb, this));
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystem_dbDup(
			ulong			pDbSystem,
			ulong			pDbToDup,
			out ulong	ppDupDb);

//-----------------------------------------------------------------------------
// setDynamicMemoryLimit
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a cache limit that is dynamically adjusted.
		/// </summary>
		/// <param name="uiCacheAdjustPercent">
		/// Percent of available memory that the cache limit is to be set to. A
		/// new cache limit is periodically recalculated based on this percentage.
		/// </param>
		/// <param name="ulCacheAdjustMin">
		/// Minimum value that the cache limit is to be set to whenever a new
		/// cache limit is calculated.
		/// </param>
		/// <param name="ulCacheAdjustMax">
		/// Maximum value that the cache limit is to be set to whenever a new cache
		/// limit is calculated.
		/// </param>
		/// <param name="ulCacheAdjustMinToLeave">
		/// This is an alternative way to specify a maximum cache limit.  If zero, 
		/// this parameter is ignored and ulCacheAdjustMax is used.  If non-zero,
		/// the maximum cache limit is calculated to be the amount of available
		/// memory minus this number - the idea being to leave a certain amount of
		/// memory for other processes to use.
		/// </param>
		public void setDynamicMemoryLimit(
			uint	uiCacheAdjustPercent,
			ulong	ulCacheAdjustMin,
			ulong	ulCacheAdjustMax,
			ulong	ulCacheAdjustMinToLeave)
		{
			RCODE	rc;

			if ((rc = xflaim_DbSystem_setDynamicMemoryLimit( m_pDbSystem, uiCacheAdjustPercent,
								ulCacheAdjustMin, ulCacheAdjustMax, ulCacheAdjustMinToLeave)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystem_setDynamicMemoryLimit(
			ulong			pDbSystem,
			uint			uiCacheAdjustPercent,
			ulong			ulCacheAdjustMin,
			ulong			ulCacheAdjustMax,
			ulong			ulCacheAdjustMinToLeave);

//-----------------------------------------------------------------------------
// setHardMemoryLimit
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a cache limit that is permanent until the next explicit call to either
		/// setHardMemoryLimit or setDynamicMemoryLimit.
		/// </summary>
		/// <param name="uiPercent">
		/// If non-zero, the new cache limit will be calculated as a
		/// percentage of either the available memory or as a percentage of the
		/// total memory on the system.  ulMin, ulMax, and ulMinToLeave are used to
		/// determine a minimum and maximum range for the new cache limit.
		/// </param>
		/// <param name="bPercentOfAvail">
		/// Only used if uiPercent is non-zero.  If true, it
		/// specifies that the percent is to be percent of available memory.  If false,
		/// the percent is the percent of total memory on the system.
		/// </param>
		/// <param name="ulMin">
		/// Only used if uiPercent is non-zero.  Specifies the minimum
		/// value that the cache limit is to be allowed to be set to.
		/// </param>
		/// <param name="ulMax">
		/// If uiPercent is non-zero, this specifies the maxmimum value
		/// that the cache limit is to be set to.  If uiPercent is zero, this specifies
		/// the new cache limit (in bytes).
		/// </param>
		/// <param name="ulMinToLeave">
		/// Only used if uiPercent is non-zero.  In that case, and this value is non-zero,
		/// this is an alternative way to specify a maximum cache limit.  If zero, this
		/// parameter is ignored and ulMax is used.  If non-zero, the maximum cache limit
		/// is calculated to be the amount of available memory (or total memory if bPercentOfAvail
		/// is false) minus this number - the idea being to leave a certain amount of memory for
		/// other processes to use.
		/// </param>
		/// <param name="bPreallocate">
		/// Flag indicating whether cache should be pre-allocated.  If true, the amount of memory
		/// specified in the new limit will be allocated immediately.  Otherwise, the memory is
		/// allocated as needed.
		/// </param>
		public void setHardMemoryLimit(
			uint		uiPercent,
			bool		bPercentOfAvail,
			ulong		ulMin,
			ulong		ulMax,
			ulong		ulMinToLeave,
			bool		bPreallocate)
		{
			RCODE	rc;

			if ((rc = xflaim_DbSystem_setHardMemoryLimit( m_pDbSystem, uiPercent,
				(int)(bPercentOfAvail ? 1 : 0), ulMin, ulMax, ulMinToLeave,
				(int)(bPreallocate ? 1 : 0))) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DbSystem_setHardMemoryLimit(
			ulong			pDbSystem,
			uint			uiPercent,
			int			bPercentOfAvail,
			ulong			ulMin,
			ulong			ulMax,
			ulong			ulMinToLeave,
			int			bPreallocate);

//-----------------------------------------------------------------------------
// getDynamicCacheSupported
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if dynamic cache limits are supported on this platform.
		/// </summary>
		/// <returns>
		/// Flag indicating whether or not dynamic cache limits are
		/// supported on this platform.
		/// </returns>
		public bool getDynamicCacheSupported()
		{
			return( xflaim_DbSystem_getDynamicCacheSupported( m_pDbSystem) != 0
						? true
						: false);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DbSystem_getDynamicCacheSupported(
			ulong			pDbSystem);

//-----------------------------------------------------------------------------
// getCacheInfo
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if dynamic cache limits are supported on this platform.
		/// </summary>
		/// <returns>
		/// Flag indicating whether or not dynamic cache limits are
		/// supported on this platform.
		/// </returns>
		public CS_XFLM_CACHE_INFO getCacheInfo()
		{
			CS_XFLM_CACHE_INFO	cacheInfo = new CS_XFLM_CACHE_INFO();

			xflaim_DbSystem_getCacheInfo( m_pDbSystem, cacheInfo);
			return( cacheInfo);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DbSystem_getCacheInfo(
			ulong					pDbSystem,
			[Out]
			CS_XFLM_CACHE_INFO	cacheInfo);

	}
}
