//------------------------------------------------------------------------------
// Desc:	Db Class
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
// $Id: Db.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * The Db class provides a number of methods that allow java applications to
 * access the XFlaim native environment, specifically, the IF_Db interface.
 */
public class Db 
{
	static
	{ 
		System.loadLibrary( "xflaimjni");
	}
	
	Db( 
		long			ref, 
		DbSystem 	dbSystem) throws XFlaimException
	{
		super();
		
		if( ref == 0)
		{
			throw new XFlaimException( -1, "No legal reference");
		}
		
		m_this = ref;

		if( dbSystem == null)
		{
			throw new XFlaimException( -1, "No legal dbSystem reference");
		}
		
		m_dbSystem = dbSystem;
	}

	/**
	 * Finalize method used to release native resources on garbage collection.
	 */	
	public void finalize()
	{
		close();
	}
	
	/**
	 *  Closes the database.
	 */	
	public void close()
	{
		// Release the native pDb!
		
		if( m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}
		
		// Remove our reference to the dbSystem so it can be released.
		
		m_dbSystem = null;
	}

	/**
	 * Starts a transaction.
	 * 
	 * @param eTransactionType The type of transaction to start (read or
	 * write).  Should be one of the members of {@link
	 * xflaim.TransactionType TransactionType}.
	 * @param iMaxLockWait Maximum lock wait time.  Specifies the amount of
	 * time to wait for lock requests occuring during the transaction to be
	 * granted.  Valid values are 0 through 255 seconds.  Zero is used to
	 * specify no-wait locks.
	 * @param iFlags Should be a logical OR'd combination of the members of
	 * the memers of {@link xflaim.TransactionFlags
	 * TransactionFlags}.
	 * @throws XFlaimException
	 */
	public void transBegin(
		int			eTransactionType,
		int			iMaxLockWait,
		int			iFlags) throws XFlaimException
	{
		_transBegin( m_this, eTransactionType, iMaxLockWait, iFlags);
	}
	
	/**
	 * Commits an existing transaction.  If no transaction is running, or the
	 * transaction commit fails, an XFlaimException exception will be thrown.
	 * @throws XFlaimException
	 */
	public void transCommit() throws XFlaimException
	{
		_transCommit( m_this);
	}
	
	/**
	 * Aborts an existing transaction.  If no transaction is running, or the
	 * transaction commit fails, an XFlaimException exception will be thrown.
	 * 
	 * @throws XFlaimException
	 */
	public void transAbort() throws XFlaimException
	{
		_transAbort( m_this);
	}
	
	/**
	 * Uses the jSearchKey to retrieve the next key from the specified
	 * index.
	 * 
	 * @param iIndex The index that is being searched
	 * @param jSearchKey The DataVector search key
	 * @param iFlags The search flags that direct how the next key will
	 * be determined.
	 * @param jFoundKey This parameter is used during subsequent calls 
	 * to keyRetrieve.  The returned DataVector is passed in as this 
	 * parameter so that it may be reused, thus preventing the unnecessary 
	 * accumulation of IF_DataVector objects in the C++ environment. 
	 */
	public void keyRetrieve(
		int				iIndex,
		DataVector		jSearchKey,
		int				iFlags,
		DataVector		jFoundKey) throws XFlaimException
	{
		long			lKey = jSearchKey.m_this;
		long			lFoundKey = (jFoundKey == null ? 0 : jFoundKey.m_this);
		
		_keyRetrieve( m_this, iIndex, lKey, iFlags, lFoundKey);
	}

	/**
	 * Creates a new document node. 
	 * @param iCollection The collection to store the new document in.
	 * @return Returns the DOMNode representing the new document.
	 * @throws XFlaimException
	 */
	 
	 public DOMNode createDocument(
	 	int			iCollection) throws XFlaimException
	{
		long 		lNewDocRef;

		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
	 		lNewDocRef =  _createDocument( m_this, iCollection);
		}
		
		return (new DOMNode( lNewDocRef, this));
	}
	
	/**
	 * Creates a new root element node. This is the root node of a document
	 * in the XFlaim database.
	 * @param iCollection
	 * @param iTag
	 * @return
	 * @throws XFlaimException
	 */
	public DOMNode createRootElement(
		int		iCollection,
		int		iTag) throws XFlaimException
	{
		long 		lNewDocRef;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lNewDocRef =  _createRootElement( m_this, iCollection, iTag);
		}
		
		return (new DOMNode( lNewDocRef, this));
	}
	
	/**
	 * Method to retrieve the first document in a specified collection.
	 * @param iCollection - The collection from which to retrieve the 
	 * first document
	 * @param jDOMNode - If this parameter is non-null, it will be assumed
	 * that it is no longer needed and will be rendered unusable upon
	 * returning from this method.
	 * @return - Returns a DOMNode which is the root node of the requested
	 * document.
	 * @throws XFlaimException
	 */
	public DOMNode getFirstDocument(
		int			iCollection,
		DOMNode		jDOMNode) throws XFlaimException
	 {
		DOMNode		jNode = null;
		long			lRef = 0;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			lRef = _getFirstDocument( m_this, iCollection, jDOMNode);
		}
	
		// If we got a reference to a native DOMNode back, let's 
		// create a new DOMNode.
		
		if (lRef != 0)
		{
			if (jDOMNode != null)
			{
				jDOMNode.setRef( lRef, this);
				jNode = jDOMNode;
			}
			else
			{
				jNode = new DOMNode( lRef, this);
			}
		}
			
		return( jNode);
	}
 
	/**
	 * Creates a new element definition in the dictionary.
	 * @param sNamespaceURI The namespace URI that this definition should be
	 * created in.  If null, the default namespace will be used.
	 * @param sElementName The name of the definition.
	 * @param iDataType The type of node this definition will represent.
	 * Should be one of the constants listed in
	 * {@link xflaim.FlmDataType FlmDataType}.
	 * @param iRequestedId If non-zero, then xflaim will try to use this
	 * number as the name ID of the new definition.
	 * @return Returns the name ID of the new definition.
	 * @throws XFlaimException
	 */
	public int createElementDef(
		String		sNamespaceURI,
		String		sElementName,
		int			iDataType,
		int			iRequestedId) throws XFlaimException
		
	{
		int	iNewNameId;
		
		// See the comments in the DOMNode::finalize() function for an
		// explanation of this call synchronized call
		
		synchronized( this)
		{
			iNewNameId = _createElementDef( m_this, sNamespaceURI,
											sElementName, iDataType,
											iRequestedId);
		}
		
		return( iNewNameId);
	}

	/**
	 * Retrieves the specified node from the specified collection
	 * @param iCollection The collection where the node is stored.
	 * @param lNodeId The ID number of the node to be retrieved
	 * @param ReusedNode Optional.  An existing instance of DOMNode who's
	 * contents will be replaced with that of the new node.  If null, a
	 * new instance will be allocated.
	 * @return Returns a DOMNode representing the retrieved node.
	 * @throws XFlaimException
	 */
	public DOMNode getNode(
		int			iCollection,
		long			lNodeId,
		DOMNode		ReusedNode) throws XFlaimException
		
	{
		long			lReusedNodeRef = 0;
		long			lNewNodeRef = 0;
		DOMNode		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.getRef();
		}
		
		// See the comments in DOMNode::finalize() for an explanation
		// of this synchronized call
		
		synchronized( this)
		{
			lNewNodeRef = _getNode( m_this, iCollection, lNodeId, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, this);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, this);
		}
		
		return( NewNode);		
	}

	/**
	 * Sets up XFlaim to perform a backup operation
	 * @param eBackupType The type of backup to perform.  Must be one of the
	 * members of {@link xflaim.FlmBackupType
	 * FlmBackupType}.
	 * @param eTransType The type of transaction in which the backup operation
	 * will take place.   Must be one of the members of
	 * {@link xflaim.TransactionType TransactionType}. 
	 * @param iMaxLockWait  Maximum lock wait time.  Specifies the amount of
	 * time to wait for lock requests occuring during the backup operation to
	 * be granted.  Valid values are 0 through 255 seconds.  Zero is used to
	 * specify no-wait locks.
	 * @param ReusedBackup Optional.  An existing instance of Backup that
	 * will be reset with the new settings.  If null, a new instance will
	 * be allocated.
	 * @return Returns an instance of Backup configured to perform the
	 * requested backup operation
	 * @throws XFlaimException
	 */
	public Backup backupBegin(
		int			eBackupType,
		int			eTransType,
		int			iMaxLockWait,
		Backup		ReusedBackup) throws XFlaimException
	{
		long 			lReusedRef = 0;
		long 			lNewRef = 0;
		Backup 		NewBackup;
		
		if (ReusedBackup != null)
		{
			lReusedRef = ReusedBackup.getRef();
		}

		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( this)
		{
			lNewRef = _backupBegin( m_this, eBackupType, eTransType,
									iMaxLockWait, lReusedRef);
		}
		
		if (ReusedBackup == null)
		{
			NewBackup = new Backup(lNewRef, this);
		}
		else
		{
			NewBackup = ReusedBackup;
			NewBackup.setRef( lNewRef, this);
		}
		
		return( NewBackup);
	}

	/**
	 * Imports an XML document into the XFlaim database.  The import requires
	 * an update transaction (TransactionType.UPDATE_TRANS). If the document
	 * cannot be imported, an XFlaimEXception exception will be thrown.
	 * @param jIStream
	 * @param iCollection
	 * @throws XFlaimException
	 */
	public void Import(
		PosIStream		jIStream,
		int				iCollection) throws XFlaimException
	{
		_import( m_this, jIStream, iCollection);
	}
	
	/**
	 * Desc:
	 */
	private native void _release(
		long				lThis);

	/**
	 * Desc:
	 */
	private native void _transBegin(
		long				lThis,
		int				iTransactionType,
		int 				iMaxlockWait,
		int 				iFlags);

	/**
	 * Desc:
	 */
	private native void _transCommit( long lThis);
	
	/**
	 * Desc:
	 */
	private native void _transAbort( long lThis);

	/**
	 * Desc:
	 */
	private native void _import(
		long				lThis,
		PosIStream		jIStream,
		int				iCollection);

	/**
	 * Desc:
	 */
 	private native long _getFirstDocument(
 		long				lThis,
 		int				iCollection,
 		DOMNode			jNode) throws XFlaimException;
 
	/**
	 * Desc:
	 */
 	private native long _getNode(
 		long				lThis,
 		int				iCollection,
 		long				lNodeId,
 		long				lpOldNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _createDocument(
		long				lThis,
		int				iCollection) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _createRootElement(
		long				lThis,
		int				iCollection,
		int				iTag) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native int _createElementDef(
		long				lThis,
		String			sNamespaceURI,
		String			sElementName,
		int				iDataType,
		int				iRequestedId) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _backupBegin(
		long				lThis,
		int				eBackupType,
		int				eTransType,
		int				iMaxLockWait,
		long				lReusedRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _keyRetrieve(
		long				lThis,
		int				iIndex,
		long				lKey,
		int				iFlags,
		long				lFoundKey);
 
	long 					m_this;
	private DbSystem 	m_dbSystem;
}
