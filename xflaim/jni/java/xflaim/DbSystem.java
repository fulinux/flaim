//------------------------------------------------------------------------------
// Desc:	Db System
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: DbSystem.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * The DbSystem class provides a number of methods that allow java 
 * applications to access the XFlaim native environment, specifically, the 
 * IF_DbSystem interface.
 */
public class DbSystem
{
	static
	{ 
		System.loadLibrary( "xflaimjni");
	}
	  
	/**
	 * Loads the appropriate native library (as determined from the system
	 * properties).
	 * 
	 * @throws XFlaimException
	 */
	public DbSystem()
			throws XFlaimException
	{
		super();
		m_this = _createDbSystem();
		_init(m_this);
	}

	public void finalize()
	{
		_exit( m_this);
		m_this = 0;
	}	

	public void dbClose()
	{
		_exit( m_this);
		m_this = 0;		
	}
		
	/**
	 * Creates a new XFlaim database.
	 * 
	 * @param sDbFileName The name of the database to create.  (Should be a
	 * filename ending in .db)
	 * @param sDataDir The directory where the database should be created.
	 * If null, the current directory will be used.
	 * @param sRflDir The directory where the roll forward log files should be
	 * stored.  If null, then they will be stored in a subdirectory under the
	 * directory containing the main database file.
	 * @param sDictFileName - The name of a file which contains dictionary
	 * definition items.  May be null.  Ignored if sDictBuf is non-null.
	 * @param sDictBuf - Contains dictionary definitions.  If null,
	 * sDictFileName is used.  If both sDictFileName and sDictBuf are null,
	 * the database is created with an empty dictionary.
	 * @param CreateOpts - An object containing several parameters that affect
	 * the creation of the database.  (For advanced users.) 
	 * @return Db reference.
	 * @throws XFlaimException
	 */
	public Db dbCreate(
		String 					sDbFileName,
		String 					sDataDir,
		String 					sRflDir,
		String 					sDictFileName,
		String 					sDictBuf,
		CREATEOPTS  			CreateOpts) throws XFlaimException
	{
	
		Db 		jDb = null;
		long 		jDb_ref;
		
		
		jDb_ref = _dbCreate( m_this, sDbFileName, sDataDir, sRflDir,
						sDictFileName, sDictBuf, CreateOpts);
		
		if( jDb_ref != 0)
		{
			jDb = new Db( jDb_ref, this);	
		}
		
		return( jDb);
	}
	
	/**
	 * Opens an existing XFlaim database.
	 * @param sDbFileName The name of the database to open.  (Should be a
	 * filename ending in .db)
	 * @param sDataDir The directory where the database should be created.
	 * If null, the current directory will be used.
	 * @param sRflDir The directory where the roll forward log files should be
	 * stored.  If null, then they will be stored in a subdirectory under the
	 * directory containing the main database file.
	 * @return Returns an instance of Db.
	 * @throws XFlaimException
	 */
	 
	public Db dbOpen(
		String				sDbFileName,
		String				sDataDir,
		String				sRflDir,
		String				sPassword,
		boolean				bAllowLimited) throws XFlaimException
	{
		Db 	jDb = null;
		long 	jDb_ref;
											
		if( (jDb_ref = _dbOpen( m_this, sDbFileName, sDataDir, 
			sRflDir, sPassword, bAllowLimited)) != 0)
		{
			jDb = new Db( jDb_ref, this);
		}
		
		return( jDb);
	}
	
	/**
	 * Removes (deletes) an XFlaim database.
	 * @param sDbFileName The name of the database to delete.  (Should be a
	 * filename ending in .db)
	 * @param sDataDir The directory where the database currently exists.
	 * If null, the current directory is assumed.
	 * @param sRflDir The directory where the roll forward log files are
	 * stored.  If null, then they are assumed to be stored in a subdirectory
	 * under the directory containing the main database file.
	 * @param bRemoveRflFiles If true, the roll forward log files will be
	 * deleted.
	 */
	public void dbRemove(
		String				sDbFileName,
		String				sDataDir,
		String				sRflDir,
		boolean				bRemoveRflFiles)
	{
		_dbRemove( m_this, sDbFileName, sDataDir, sRflDir, bRemoveRflFiles);
	}
	
	/**
	 * Restores a previously backed up database.  <code>sBackupPath</code> and 
	 * <code> RestoreClient</code> are mutually exclusive.  If
	 * <code>RestoreClient</code> is null, then an instance of
	 * <code>DefaultRestoreClient</code> will be created and
	 * <code>sBackupPath</code> passed into its constructor.  If <code>
	 * RestoreClient</code> is non-null, <code>sBackupPath</code> is ignored.
	 * @param sDbPath The name of the database to create.  (Should be a
	 * filename ending in .db)
	 * @param sDataDir Optional.  The directory where the new data files will
	 * be stored.  If null, then they will be stored in the same directory as
	 * the .db file.
	 * @param sRflDir Optional.  The directory where RFL files will be stored.
	 * If null, then they will be stored in a subdirectory under the directory
	 * containing the .db file.
	 * @param sBackupPath Optional.  The path to the backup files.
	 * @param RestoreClient Optional.  An object implementing the
	 * {@link RestoreClient RestoreClient} interface
	 * @param RestoreStatus Optional.  An object implementing the
	 * {@link RestoreStatus RestoreStatus} interface
	 * @throws XFlaimException
	 */
	public void dbRestore(
		String			sDbPath,
		String			sDataDir,
		String			sRflDir,
		String			sBackupPath,
		String			sPassword,
		RestoreClient	RestoreClient,
		RestoreStatus	RestoreStatus) throws XFlaimException
	{
		RestoreClient	Client;
		
		if (RestoreClient != null)
		{
			Client = RestoreClient;
		}
		else
		{
			Client = new DefaultRestoreClient( sBackupPath);
		}
		
		_dbRestore( m_this, sDbPath, sDataDir, sRflDir, sBackupPath,
				sPassword, Client, RestoreStatus);
	}


	/**
	 * Opens a buffered input stream.
	 * @param sBuffer
	 * @return Returns an instance of PosIStream.
	 */
	public PosIStream openBufferIStream(
		String				sBuffer) throws XFlaimException
	{
		PosIStream	jPosIStream = null;
		long					lRef = 0;

		lRef = _openBufferIStream( m_this, sBuffer);
		
		if (lRef != 0)
		{
			jPosIStream = new PosIStream( lRef,  sBuffer, this);
		}
		
		return( jPosIStream);
	}

	/**
	 * Opens a file to be used as an input stream.
	 * @param sPath The pathname of the file to be opened.
	 * @return Returns an instance of PosIStream.
	 * @throws XFlaimException
	 */
	public PosIStream openFileIStream( String sPath) throws XFlaimException
	{
		PosIStream		jIStream = null;
		long					lRef = 0;
		
		lRef = _openFileIStream(
								m_this,
								sPath);

		if (lRef != 0)
		{
			jIStream = new PosIStream( lRef, this);		
		}
		
		return( jIStream);
	}

	/**
	 * Creates and returns a DataVector object to be used when searching
	 * indexes.
	 * @return DataVector
	 */	
	public DataVector createJDataVector() throws XFlaimException
	{
		DataVector		jDataVector = null;
		long				lRef = 0;
		
		lRef = _createJDataVector(m_this);
							
		if (lRef != 0)
		{
			jDataVector = new DataVector(lRef, this);
		}
		
		return jDataVector;
	}

	/**
	 * Peforms an integrity check on the specified database.
	 * @param sDbFileName The name of the database to be checked.  (Should be a
	 * filename ending in .db)
	 * @param sDataDir Optional.  The directory where the data files are
	 * stored.  If null, then XFlaim will assume that they are stored in the
	 * same directory as the .db file. 
	 * @param sRflDir Optional.  The directory where RFL files are stored.
	 * If null, then XFlaim will assume that they are stored in a subdirectory
	 * under the directory containing the .db file.
	 * @param iFlags Flags that control exactly what the operation checks.
	 * Should be a logical OR of the members of
	 * {@link xflaim.DbCheckFlags DbCheckFlags}.
	 * @param Status Optional.  If non-null, then XFlaim will call member
	 * functions to report progress of the check and report any errors that
	 * are found. 
	 * @return Returns an instance of DbInfo containing data on the physical
	 * structure of the database. 
	 * @throws XFlaimException
	 */
	public DbInfo dbCheck(
		String			sDbFileName,
		String			sDataDir,
		String			sRflDir,
		String			sPassword,
		int				iFlags,
		DbCheckStatus	Status) throws XFlaimException
	{
		 long	lRef = _dbCheck( m_this, sDbFileName, sDataDir, sRflDir,
		 						 sPassword, iFlags, Status);
		 return new DbInfo( lRef);
	}
	
	/**
	 * Makes a copy of an existing database.
	 * @param sSrcDbName The name of the existing database.  (Should be a
	 * filename ending in .db)
	 * @param sSrcDataDir Optional.  The directory where the data files are
	 * stored.  If null, then XFlaim will assume that they are stored in the
	 * same directory as the .db file.
	 * @param sSrcRflDir Optional.  The directory where RFL files are stored.
	 * If null, then XFlaim will assume that they are stored in a subdirectory
	 * under the directory containing the .db file.
	 * @param sDestDbName The name for the new database.
	 * @param sDestDataDir Optional.  The directory where the data files for
	 * the new database will be stored.
	 * @param sDestRflDir Optional.  The directory where the RFL files for the
	 * new database will be stored.
	 * @param Status Optional.  If non-null, then <code>Status.dbCopyStatus
	 * </code> will be called periodically.
	 * @throws XFlaimException
	 */
	public void dbCopy(
		String			sSrcDbName,
		String			sSrcDataDir,
		String			sSrcRflDir,
		String			sDestDbName,
		String			sDestDataDir,
		String			sDestRflDir,
		DbCopyStatus	Status) throws XFlaimException
	{
		_dbCopy( m_this, sSrcDbName, sSrcDataDir, sSrcRflDir,
				 sDestDbName, sDestDataDir, sDestRflDir, Status);
	}

	/**
	 * Renames a database.
	 * @param sDbName The name of the database to be renamed.  (Should be a
	 * filename ending in .db)
	 * @param sDataDir Optional.  The directory where the data files are
	 * stored.  If null, then XFlaim will assume that they are stored in the
	 * same directory as the .db file.
	 * @param sRflDir Optional.  The directory where RFL files are stored.
	 * If null, then XFlaim will assume that they are stored in a subdirectory
	 * under the directory containing the .db file.
	 * @param sNewDbName The new name for the database.
	 * @param bOverwriteDestOk If true, then if the database specified in
	 * sNewDbName already exists, it will be overwritten.
	 * @param Status Optional.  If non-null, then <code>Status.dbRenameStatus
	 * </code> will be called as every file is renamed. 
	 * @throws XFlaimException
	 */
	public void dbRename(
		String				sDbName,
		String				sDataDir,
		String				sRflDir,
		String				sNewDbName,
		boolean				bOverwriteDestOk,
		DbRenameStatus		Status) throws XFlaimException
	{
		_dbRename( m_this, sDbName, sDataDir, sRflDir, sNewDbName,
				   bOverwriteDestOk, Status);
	}

	/**
	 * Desc:
	 */
	private native long _createDbSystem();
	
	/**
	 * Desc:
	 */
	private native void _init( long lThis);
	
	/**
	 * Desc:
	 */
	private native void _exit( long lThis);

	/**
	 * Desc:
	 */
	private native long _dbCreate(
		long					lThis,
		String 				DbFileName,
		String 				DataDir,
		String 				RflDir,
		String 				DictFileName,
		String 				DictBuf,
		CREATEOPTS  		CreateOpts);

	private native long _dbOpen(
		long					lThis,
		String				DbFileName,
		String				DataDir,
		String				RflDir,
		String				Password,
		boolean				bAllowLimited);

	/**
	 * Desc:
	 */
	private native void _dbRemove(
		long					lThis,
		String				DbFileName,
		String				DataDir,
		String				RflDir,
		boolean				bRemoveRflFiles);

	/**
	 * Desc:
	 */
	private native long _dbCheck(
		long					lThis,
		String				sDbFileName,
		String				sDataDir,
		String				sRflDir,
		String				sPassword,
		int					iFlags,
		DbCheckStatus		Status) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _dbCopy(
		long					lThis,
		String				sSrcDbName,
		String				sSrcDataDir,
		String				sSrcRflDir,
		String				sDestDbName,
		String				sDestDataDir,
		String				sDestRflDir,
		DbCopyStatus		Status) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _dbRestore(
		long					lThis,
		String				sDbPath,
		String				sDataDir,
		String				sRflDir,
		String				sBackupPath,
		String				sPassword,
		RestoreClient		RestoreClient,
		RestoreStatus		RestoreStatus) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native void _dbRename(
		long					lThis,
		String				sDbName,
		String				sDataDir,
		String				sRflDir,
		String				sNewDbName,
		boolean				bOverwriteDestOk,
		DbRenameStatus		Status) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _openBufferIStream(
		long					lThis,
		String				sBuffer) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _openFileIStream(
		long					lThis,
		String				sPath);

	/**
	 * Desc:
	 */
	private native long _createJDataVector(
		long					lRef);

	private long			m_this;
}
