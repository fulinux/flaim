//------------------------------------------------------------------------------
// Desc:	DOM Node
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
// $Id: DOMNode.java 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * This class encapsulates the XFlaim F_DOMNode interface.
 */
public class DOMNode 
{
	
	/**
	 * Desc:
	 */
	DOMNode( 
		long			lThis,
		Db 			jdb) throws XFlaimException
	{
		if (lThis == 0)
		{
			throw new XFlaimException( -1, "No legal reference to a DOMNode");
		}
		
		m_this = lThis;
		
		if (jdb == null)
		{
			throw new XFlaimException( -1, "No legal jDb reference");
		}
		
		m_jdb = jdb;
	}
	
	/**
	 * Desc:
	 */
	protected void finalize()
	{
		// The F_DOMNode and F_Db classes are not thread-safe.  The proper way
		// of using XFlaim is to get a new instance of Db for each thread.
		// Unfortunately, the garbage collector runs in its own thread.  This
		// leads to a potential race condition down in the C++ code when one
		// thread tries to create an already existing node (which results in a
		// call to F_DOMNode::AddRef()) and the GC tries to destroy the same
		// node (which results in a call to F_DOMNode::Release()).
		// We protect against this by synchronizing against the instance of
		// Db.  Note that we are not protecting any of the accesses to the
		// node; only creating and destroying.  DOMNode and Db are still
		// not completely thread-safe.
		
		synchronized( m_jdb)
		{
			// Release the associated DOMNode.
			
			if (m_this != 0)
			{
				_release( m_this);
			}
		}
		
		// Free our reference to the Db object.
		
		m_jdb = null;
	}
	
	/**
	 * Desc:
	 */
	public void release()
	{
		synchronized( m_jdb)
		{
			if (m_this != 0)
			{
				_release( m_this);
			}
		}
		
		m_jdb = null;
	}
	
	/**
	 * Creates a new DOM node and inserts it into the database in the
	 * specified position relative to the current node.  An existing
	 * DOMNode object can optionally be passed in, and it will be reused
	 * instead of a new object being allocated.
	 * @param iNodeType An integer representing the type of node to create.
	 * (Use the constants in {@link xflaim.FlmDomNodeType FlmDomNodeType}.)
	 * @param iNameId The dictionary tag number that represents the node name.
	 * This value must exist in the dictionary before it can be used here.  The
	 * value may be one of the predefined ones, or it may be created with
	 * {@link Db#createElementDef Db::createElementDef}.
	 * @param iInsertLoc An integer representing the relative position to insert
	 * the new node.  (Use the constants in 
	 * {@link xflaim.FlmInsertLoc FlmInsertLoc}.)
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode createNode(
		int			iNodeType,
		int			iNameId,
		int			iInsertLoc,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createNode( m_this, m_jdb.m_this, iNodeType, iNameId,
									   iInsertLoc, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
		
		return( NewNode);
	}

	/**
	 *  Removes this node as well as all of it's descendants from the database.
	 * @throws XFlaimException
	 */
	public void deleteNode() throws XFlaimException
	{
		_deleteNode( m_this, m_jdb.m_this);		
	}
	
	/**
	 * Removes the children of the current node from the database.
	 * @throws XFlaimException
	 */
	public void deleteChildren() throws XFlaimException
	{
		_deleteChildren( m_this, m_jdb.m_this);	
	}

	/**
	 * Checks the type of node.  Returned value will be one of those listed in
	 * {@link xflaim.FlmDomNodeType FlmDomNodeType}.
	 * @return Returns the type of the current node.
	 */
	public int getNodeType()
	{
		return _getNodeType( m_this);
	}

	/**
 	* Tests the current node for the ability to hold data.
 	* @return Returns true if this is a node type that can have data 
	* associated with it.
 	*/
	public boolean canHaveData()
	{
		return _canHaveData( m_this);
	}

	/**
	 * Creates a new attribute node assigned to the current node.  Note that
	 * some nodes are not allowed to have attributes.
	 * @param iNameId The dictionary tag number that represents the node name.
	 * This value must exist in the dictionary before it can be used here.  The
	 * value may be one of the predefined ones, or it may be created with
	 * {@link Db#createElementDef Db::createElementDef}.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 */
	public DOMNode createAttribute(
		int			iNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}

		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createAttribute( m_this, m_jdb.m_this, iNameId,
											lReusedNodeRef);
		}

		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode);
	}

	/**
	 * Retrieves the first attribute node associated with the current node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getFirstAttribute(
		DOMNode ReusedNode) throws XFlaimException
	{
		long 		lReusedNodeRef = 0;
		long 		lNewNodeRef = 0;
		DOMNode 	NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}

		// See the comment in the finalize function for an explanation of
		// this synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getFirstAttribute( m_this, 
									m_jdb.m_this, lReusedNodeRef);
		}

		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode);
	}
	
	/**
	 * Retrieves the last attribute node associated with the current node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode.
	 * @throws XFlaimException
	 */
	public DOMNode getLastAttribute(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}

		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getLastAttribute( m_this, m_jdb.m_this, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode);
	}
	
	/**
	 * Retrieves the requested attribute node associated with this node.
	 * @param iAttributeId The dictionary tag number of the requested
	 * attribute.  This value must exist in the dictionary before it can be
	 * used here.  The value may be one of the predefined ones, or it may be
	 * created with {@link Db#createElementDef Db::createElementDef}.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getAttribute(
		int			iAttributeId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}

		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getAttribute( m_this, m_jdb.m_this, iAttributeId,
										 lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}

		return( NewNode); 
	}
	
	/**
	 * Removes the specified attribute node from the current node.
	 * @param iAttributeId The dictionary tag number representing the
	 * attribute node to be deleted.
	 * @throws XFlaimException
	 */
	public void deleteAttribute(
		int		iAttributeId) throws XFlaimException
	{
		_deleteAttribute( m_this, m_jdb.m_this, iAttributeId);
	}
  
  	/**
  	 * Looks through the list of attributes for the one specified in iNameId.
  	 * Note that this function's semantics differ from its C++ counterpart.
  	 * @return Returns true if the attribute was found; false otherwise.
  	 * @throws XFlaimException
  	 */
	public boolean hasAttribute(
		int		iNameId)  throws XFlaimException
	{
		return _hasAttribute( m_this, m_jdb.m_this, iNameId);
	}
	
	/**
	 * Tests to see if this node as any attributes associated with it.
	 * @return Returns true if the node has any attributes.
	 * @throws XFlaimException
	 */
	public boolean hasAttributes() throws XFlaimException
	{
		return _hasAttributes( m_this, m_jdb.m_this);
	}
	
	/**
	 * Tests to see if this node has a next sibling.
	 * @return Returns true if this node has a next sibling.
	 * @throws XFlaimException
	 */
	public boolean hasNextSibling() throws XFlaimException
	{
		return _hasNextSibling( m_this, m_jdb.m_this);
	}

	/**
	 * Tests to see if this node has a previous sibling.
	 * @return Returns true if this node has a previous sibling.
	 * @throws XFlaimException
	 */
	public boolean hasPreviousSibling() throws XFlaimException
	{
		return _hasPreviousSibling( m_this, m_jdb.m_this);
	}

	/**
	 * Tests to see if this node has any child nodes.
	 * @return Returns true if this node has any children.
	 * @throws XFlaimException
	 */
	public boolean hasChildren() throws XFlaimException
	{
		return _hasChildren( m_this, m_jdb.m_this);
	}	
	/**
	 * Tests to see if this node is an attribute node that defines a namespace.
	 * @return Returns true if this node is an attribute node that defines a
	 * namespace and false if this node not an attribute node or it does not
	 * define a namespace. 
	 * @throws XFlaimException
	 */
	public boolean isNamespaceDecl() throws XFlaimException
	{
		return _isNamespaceDecl( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's parent.
	 * @return Returns the parent node's node ID.
	 * @throws XFlaimException
	 */	
	public long getParentId() throws XFlaimException
	{
		return _getParentId( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for this node.
	 * @return Returns the node ID.
	 */
	public long getNodeId()
	{
		return _getNodeId( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for the root node in the document
	 * @return Returns the root node's node ID.
	 * @throws XFlaimException
	 */
	public long getDocumentId() throws XFlaimException
	{
		return _getDocumentId( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for this node's previous sibling. 
	 * @return Returns the previous sibling node's node ID.
	 * @throws XFlaimException
	 */
	public long getPrevSibId() throws XFlaimException
	{
		return _getPrevSibId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's next sibling.
	 * @return Returns the next sibling node's node ID.
	 * @throws XFlaimException
	 */
	public long getNextsibId() throws XFlaimException
	{
		return _getNextSibId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's first child. 
	 * @return Returns the first child node's node ID.
	 * @throws XFlaimException
	 */	
	public long getFirstChildId() throws XFlaimException
	{
		return _getFirstChildId(  m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the node ID for this node's next child. 
	 * @return Returns the next child node's node ID.
	 * @throws XFlaimException
	 */	
	public long getLastChildId() throws XFlaimException
	{
		return _getLastChildId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's first attribute.
	 * @return Returns the first attribute node's node ID.
	 * @throws XFlaimException
	 */
	public long getFirstAttrId() throws XFlaimException
	{
		return _getFirstAttrId(  m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the node ID for this node's last attribute 
	 * @return Returns the last attribute node's node ID.
	 * @throws XFlaimException
	 */
	public long getLastAttrId() throws XFlaimException
	{
		return _getLastAttrId(  m_this, m_jdb.m_this);
	}	
	
	/**
	 * Retrieves this node's name ID. 
	 * @return Returns the name ID for this node.
	 * @throws XFlaimException
	 */
	public int getNameId() throws XFlaimException
	{
		return _getNameId( m_this, m_jdb.m_this);
	}

	/**
	 * Assigns a 64-bit value to this node.
	 * @param lValue The value to be assigned.
	 * @throws XFlaimException
	 */
	public void setLong(
		long			lValue) throws XFlaimException
	{
		_setLong( m_this, m_jdb.m_this, lValue);	
	}
	
	/**
	 * Assigns a text string to this node.  Existing text is either
	 * overwritten or has the new text appended to it.  See the
	 * explanation for the bLast parameter.
	 * @param sValue The text to be assigned
	 * @param bLast Specifies whether sValue is the last text to be
	 * appended to this node.  If false, then another call to setString
	 * is expected, and the new text will be appended to the text currently
	 * stored in this node.  If true, then no more text is expected and 
	 * another call to setString will overwrite the what is currently
	 * stored in this node.
	 * @throws XFlaimException
	 */
	public void setString(
		String		sValue,
		boolean		bLast) throws XFlaimException
	{
		_setString( m_this, m_jdb.m_this, sValue, bLast);
	}

	/**
	 * Assigns or appends a text string to this node.  This function is
	 * equivalent to setString( sValue, true).
	 * @param sValue The text to be assigned.
	 * @throws XFlaimException
	 * @see #setString( String, boolean)
	 */
	public void setString(
		String		sValue) throws XFlaimException
	{
		_setString( m_this, m_jdb.m_this, sValue, true);			
	}
	
	/**
	 * Assigns a piece of binary data to this node.
	 * @param Value An array of bytes to be stored in this node.
	 * @throws XFlaimException
	 */
	public void setBinary(
		byte[] 		Value) throws XFlaimException
	{
		_setBinary( m_this, m_jdb.m_this, Value);
	}

	/**
	 * Retrieves the amount of memory occupied by the value of this node. 
	 * @return Returns the length of the data stored in the node (in bytes).
	 * @throws XFlaimException
	 */
	public long getDataLength() throws XFlaimException
	{
		return _getDataLength( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the type of the value stored in this node.  The value will
	 * be one of those listed in 
	 * {@link xflaim.FlmDataType FlmDataType}.
	 * @return Returns the type of the value stored in this node.
	 * @throws XFlaimException
	 */
	public int getDataType() throws XFlaimException
	{
		return _getDataType( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the value stored in this node as a long. 
	 * @return Returns the value stored in the node.
	 * @throws XFlaimException
	 */
	public long getLong() throws XFlaimException
	{
		return _getLong( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves a string representation of the value stored in this node. 
	 * @return Returns the value stored in the node.
	 * @throws XFlaimException
	 */	
	public String getString() throws XFlaimException
	{
		return _getString( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the number of unicode characters a string representation of
	 * the node's value would occupy.
	 * @return Returns the length of a string representation of this node's
	 * data.
	 * @throws XFlaimException
	 */
	public int getStringLen() throws XFlaimException
	{
		return _getStringLen( m_this, m_jdb.m_this);		
	}

	/**
	 * Retrieves the value of the node as raw data.
	 * @return Returns a byte array containing the value of this node. 
	 * @throws XFlaimException
	 */
	public byte[] getBinary() throws XFlaimException
	{
		return _getBinary( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the parent of the current node. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode 
	 * @throws XFlaimException
	 */
	public DOMNode getParentNode(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
		
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		synchronized( m_jdb)
		{
			lNewNodeRef = _getParentNode( m_this, m_jdb.m_this, lReusedNodeRef);
		}
		
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
		
		return NewNode;
	}

	/**
	 * Retrieves the first node in the current node's list of child nodes.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getFirstChild(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// call synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getFirstChild( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the last node in the current node's list of child nodes. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode;
	 * @throws XFlaimException
	 */
	public DOMNode getLastChild(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getLastChild( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the first instance of the specified type of node from the
	 * current node's list of child nodes.
	 * @param eNodeType The value representing the node type.  
	 * (Use the constants in FlmDomNodeType.)
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode;
	 * @throws XFlaimException
	 */
	public DOMNode getChild(
		int			eNodeType,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getChild( m_this, m_jdb.m_this, 
									eNodeType, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the specified element node from the current node's 
	 * list of child nodes.
	 * @param iNameId The name ID for the desired node
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an  instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getChildElement(
		int			iNameId,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getChildElement( m_this, m_jdb.m_this, 
									iNameId, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the specified element node from the current node's 
	 * list of sibling nodes.
	 * @param iNameId The name ID of the desired node.
	 * @param bNext If true, will search forward following each node's
	 * "next_sibling" link; if false, will follow each node's "prev_sibling"
	 * link. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getSiblingElement(
		int			iNameId,
		boolean		bNext,
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getSiblingElement( m_this, m_jdb.m_this, iNameId,
											  bNext, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;		
	}

	/**
	 * Retrieve's the previous node from the current node's list of 
	 * siblings nodes. 
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getPreviousSibling(
		DOMNode 		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getPreviousSibling( m_this, m_jdb.m_this, 
									lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}
	
	/**
	 * Retrieves the next node from the current node's list of sibling nodes.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getNextSibling(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getNextSibling( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the previous document node.  The current node must be a root
	 * node or a document node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getPreviousDocument(
		DOMNode		ReusedNode) throws XFlaimException
	{
		long 			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getPreviousDocument( m_this, 
									m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the next document node.  The current node must be a root
	 * node or a document node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)
	 * @return Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode getNextDocument(
		DOMNode 		ReusedNode) throws XFlaimException
	{
		long			lReusedNodeRef = 0;
		long 			lNewNodeRef = 0;
		DOMNode 		NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getNextDocument( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the namespace prefix for this node
	 * @return Returns a string containing this node's namespace prefix
	 * @throws XFlaimException
	 */
	public String getPrefix () throws XFlaimException
	{
		return _getPrefix( m_this, m_jdb.m_this);
	}

	/**
	 * Retrieves the namespace URI that this node's name belongs to.
	 * @return Returns the namespace URI
	 * @throws XFlaimException
	 */
	public String getNamespaceURI() throws XFlaimException
	{
		return _getNamespaceURI( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the name of this node, without the namespace prefix.
	 * @return Returns unprefixed element or attribute name.
	 * @throws XFlaimException
	 */
	public String getLocalName() throws XFlaimException
	{
		return _getLocalName( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the fully qualified name (namespace prefix plus local
	 * name) for this element or attribute.
	 * @return Returns the fully qualified element or attribute name.
	 * @throws XFlaimException
	 */
	public String getQualifiedName() throws XFlaimException
	{
		return _getQualifiedName( m_this, m_jdb.m_this);
	}
	
	/**
	 * Retrieves the collection that this node is stored in.
	 * @return Returns the collection number.
	 * @throws XFlaimException
	 */
	public int getCollection() throws XFlaimException
	{
		return _getCollection( m_this, m_jdb.m_this);
	}

	/**
	 * Creates an annotation node and assignes it to the current node
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return  Returns an instance of DOMNode
	 * @throws XFlaimException
	 */
	public DOMNode createAnnotation(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 				lReusedNodeRef = 0;
		long 				lNewNodeRef = 0;
		DOMNode 			NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _createAnnotation( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;
	}

	/**
	 * Retrieves the annotation node assigned to this node.
	 * @param ReusedNode An instance of DOMNode which is no longer needed and
	 * can be reassigned to point to different data in the database.  (Reusing
	 * DOMNode objects is encouraged as it saves the system from allocating
	 * and freeing memory for each object.)  Can be null, if no instances are
	 * available to be reused.
	 * @return returns the annotation node assigned to this node
	 * @throws XFlaimException
	 */
	public DOMNode getAnnotation(
		DOMNode			ReusedNode) throws XFlaimException
	{
		long 				lReusedNodeRef = 0;
		long 				lNewNodeRef = 0;
		DOMNode 			NewNode;
		
		if (ReusedNode != null)
		{
			lReusedNodeRef = ReusedNode.m_this;
		}
			
		// See to comments in the finalize function for an explanation of this
		// synchronized call
		
		synchronized( m_jdb)
		{
			lNewNodeRef = _getAnnotation( m_this, m_jdb.m_this, lReusedNodeRef);
		}
			
		if (ReusedNode == null)
		{
			NewNode = new DOMNode(lNewNodeRef, m_jdb);
		}
		else
		{
			NewNode=ReusedNode;
			NewNode.setRef( lNewNodeRef, m_jdb);
		}
			
		return NewNode;		
	}

	/**
	 * Checks to see if this node has an annotation
	 * @return Returns true if the current node has an annotation
	 * @throws XFlaimException
	 */
	public boolean hasAnnotation() throws XFlaimException
	{
		return _hasAnnotation( m_this, m_jdb.m_this);
	}

	/**
	 * Reassigns the object to "point" to a new F_DOMNode instance and a new
	 * Db.  Called by any of the member functions that take a ReusuedNode
	 * parameter.  Shouldn't be called by outsiders, so it's not public, but
	 * it must be callable for other instances of this class.  (It's also
	 * called by Db.getNode)
	 *
	 * NOTE:  This function does not result in a call to F_DOMNode::Release()
	 * because that is done by the native code when the F_DOMNode object is 
	 * reused.  Calling setRef() in any case except after a DOM node has been
	 * reused will result in a memory leak on the native side!
	*/
	void setRef(
		long	lDomNodeRef,
		Db	jdb)
	{
		m_this = lDomNodeRef;
		m_jdb = jdb;
	}

	/**
	 * Desc:
	 */
	long getRef()
	{
		return m_this;
	}
	
	/**
	 * Desc:
	 */
	Db getJdb()
	{
		return m_jdb;
	}
	
	/**
	 * Desc:
	 */
	 private native long _createNode(
		 long		lThis,
		 long		lpDbRef,
		 int		iNodeType,
		 int		iNameId,
		 int		iInsertLoc,
		 long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _deleteNode(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		 
	/**
	 * Desc:
	 */
	private native void _deleteChildren(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native int _getNodeType(
		long		lThis);

	/**
	 * Desc:
	 */
	private native boolean _canHaveData(
		long 		lThis);
		
	/**
	 * Desc:
	 */
	private native long _createAttribute(
		long		lThis,
		long		lpDbRef,
		int		iNameId,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getFirstAttribute(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getLastAttribute(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getAttribute(
		long		lThis,
		long		lpDbRef,
		int		iAttributeId,
		long		lReusedNodeRef) throws XFlaimException;
	
	/**
	 * Desc:
	 */
	private native void _deleteAttribute(
		long		lThis,
		long		lpDbRef,
		int		iAttributeId) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native boolean _hasAttribute(
		long		lThis,
		long		lpDbRef,
		int		iAttributeId) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native boolean _hasAttributes(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native boolean _hasChildren(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
	
	/**
	 * Desc:
	 */
	private native boolean _hasNextSibling(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
	
	/**
	 * Desc:
	 */
	private native boolean _hasPreviousSibling(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
	
	/**
	 * Desc:
	 */
	private native boolean _isNamespaceDecl(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getParentNode(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getFirstChild(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getLastChild(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getChild(
		long		lThis,
		long		lpDbRef,
		int		iNodeType,
		long		lReusedNodeRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getPreviousSibling(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;
			
	/**
	 * Desc:
	 */
	private native long _getNextSibling
	(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getPreviousDocument
	(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getNextDocument(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native String _getPrefix(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getChildElement(
		long		lThis,
		long		lpDbRef,
		int		iNameId,
		long		lReusedNodeRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getSiblingElement(
		long		lThis,
		long		lpDbRef,
		int		iNameId,
		boolean	bNext,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getParentId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getNodeId(
		long		lThis,
		long		lpDbRef);
		
	/**
	 * Desc:
	 */
	private native long _getDocumentId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getPrevSibId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getNextSibId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getFirstChildId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getLastChildId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getFirstAttrId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getLastAttrId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native int _getNameId(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native String _getNamespaceURI(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native String _getLocalName(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native String _getQualifiedName(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native int _getCollection(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		

	/**
	 * Desc:
	 */
	private native long _getLong(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native String _getString(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native int _getStringLen(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native int _getDataType(
		long		lThis,
		long		lpDbRef) throws XFlaimException;
		
	/**
	 * Desc:
	 */
	private native long _getDataLength(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native byte[] _getBinary(
		long		lThis,
		long		lpDbRef) throws XFlaimException; 

	/**
	 * Desc:
	 */
	private native void _setLong(
		long		lThis,
		long		lpDbRef,
		long		lValue) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _setString(
		long		lThis,
		long		lpDbRef,
		String	sValue,
		boolean	bLast) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _setBinary(
		long		lThis,
		long		lpDbRef,
		byte[]	Value) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _createAnnotation(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native long _getAnnotation(
		long		lThis,
		long		lpDbRef,
		long		lReusedNodeRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native boolean _hasAnnotation(
		long		lThis,
		long		lpDbRef) throws XFlaimException;

	/**
	 * Desc:
	 */
	private native void _release(
		long		lThis);
		
	public long getThis()
	{
		return m_this;
	}
	
	private long	m_this;
	private Db		m_jdb;
}
