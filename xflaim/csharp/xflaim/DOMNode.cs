//------------------------------------------------------------------------------
// Desc:	DOM Node
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
	/// Data types supported in an XFLAIM database.
	/// IMPORTANT NOTE: These should be kept in sync with data types defined
	/// in xflaim.h
	/// </summary>
	public enum FlmDataType : uint
	{
		/// <summary>No data may be stored with a node of this type</summary>
		XFLM_NODATA_TYPE			= 0,
		/// <summary>String data - UTF8 - unicode 16 supported</summary>
		XFLM_TEXT_TYPE				= 1,
		/// <summary>Integer numbers - 64 bit signed and unsigned supported</summary>
		XFLM_NUMBER_TYPE			= 2,
		/// <summary>Binary data</summary>
		XFLM_BINARY_TYPE			= 3
	}

	/// <summary>
	/// DOM Node types
	/// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	/// definitions in xflaim.h
	/// </summary>
	public enum eDomNodeType : uint
	{
		/// <summary>Invalid Node</summary>
		INVALID_NODE =							0x00,
		/// <summary>Document Node</summary>
		DOCUMENT_NODE =						0x01,
		/// <summary>Element Node</summary>
		ELEMENT_NODE =							0x02,
		/// <summary>Data Node</summary>
		DATA_NODE =								0x03,
		/// <summary>Comment Node</summary>
		COMMENT_NODE =							0x04,
		/// <summary>CDATA Section Node</summary>
		CDATA_SECTION_NODE =					0x05,
		/// <summary>Annotation Node</summary>
		ANNOTATION_NODE =						0x06,
		/// <summary>Processing Instruction Node</summary>
		PROCESSING_INSTRUCTION_NODE =		0x07,
		/// <summary>Attribute Node</summary>
		ATTRIBUTE_NODE =						0x08,
		/// <summary>Any Node Type</summary>
		ANY_NODE_TYPE =						0xFFFF
	}

	/// <summary>
	/// Node insert locations - relative to another node.
	/// IMPORTANT NOTE: These need to be kept in sync with the corresponding
	/// definitions in xflaim.h
	/// </summary>
	public enum eNodeInsertLoc : uint
	{
		/// <summary>Insert node as root node of document</summary>
		XFLM_ROOT = 0,
		/// <summary>Insert node as first child of reference node</summary>
		XFLM_FIRST_CHILD,
		/// <summary>Insert node as last child of reference node</summary>
		XFLM_LAST_CHILD,
		/// <summary>Insert node as previous sibling of reference node</summary>
		XFLM_PREV_SIB,
		/// <summary>Insert node as next sibling of reference node</summary>
		XFLM_NEXT_SIB,
		/// <summary>Insert node as attribute of reference node</summary>
		XFLM_ATTRIBUTE
	}

	/// <remarks>
	/// The DOMNode class provides a number of methods that allow C#
	/// applications to access DOM nodes in XML documents.
	/// </remarks>
	public class DOMNode
	{
		private ulong 		m_pNode;			// Pointer to IF_DOMNode object in unmanaged space
		private Db			m_db;

		/// <summary>
		/// DOMNode constructor.
		/// </summary>
		/// <param name="pNode">
		/// Reference to an IF_DOMNode object.
		/// </param>
		/// <param name="db">
		/// Db object that this DOMNode object is associated with.
		/// </param>
		internal DOMNode(
			ulong		pNode,
			Db			db)
		{
			if (pNode == 0)
			{
				throw new XFlaimException( "Invalid IF_DOMNode reference");
			}
			
			m_pNode = pNode;

			if (db == null)
			{
				throw new XFlaimException( "Invalid Db reference");
			}
			
			m_db = db;
			
			// Must call something inside of Db.  Otherwise, the
			// m_db object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_db.getDb() == 0)
			{
				throw new XFlaimException( "Invalid Db.IF_Db object");
			}
		}

		/// <summary>
		/// Set the IF_DOMNode pointer inside this object.  NOTE: We deliberately
		/// do NOT release the m_pNode in this case, because it will already have
		/// been released by the caller.  Usually, the caller has made a call into
		/// the native C++ code that will have released this pointer if it was
		/// successful.
		/// </summary>
		/// <param name="pNode">
		/// Reference to an IF_DOMNode object.
		/// </param>
		/// <param name="db">
		/// Db object that this DOMNode object is associated with.
		/// </param>
		internal void setNodePtr(
			ulong		pNode,
			Db			db)
		{
			m_pNode = pNode;
			m_db = db;
		}

		/// <summary>
		/// Destructor.
		/// </summary>
		~DOMNode()
		{
			close();
		}

		/// <summary>
		/// Return the pointer to the IF_DOMNode object.
		/// </summary>
		/// <returns>Returns a pointer to the IF_DOMNode object.</returns>
		internal ulong getNode()
		{
			return( m_pNode);
		}

		/// <summary>
		/// Close this DOM node.
		/// </summary>
		public void close()
		{
			// Release the native pNode!
		
			if (m_pNode != 0)
			{
				xflaim_DOMNode_Release( m_pNode);
				m_pNode = 0;
			}
		
			// Remove our reference to the db so it can be released.
		
			m_db = null;
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DOMNode_Release(
			ulong	pNode);

//-----------------------------------------------------------------------------
// createNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new DOM node and inserts it into the database in the
		/// specified position relative to the current node.  An existing
		/// DOMNode object can optionally be passed in, and it will be reused
		/// instead of a new object being allocated.
		/// </summary>
		/// <param name="eNodeType">
		/// Type of node to create.
		/// </param>
		/// <param name="uiNameId">
		/// The dictionary tag number that represents the node name. This name ID
		/// must exist in the dictionary before it can be used here.  The value
		/// may be one of the predefined ones, or it may be created by calling
		/// <see cref="Db.createElementDef"/>.
		/// </param>
		/// <param name="eInsertLoc">
		/// The relative position to insert the new node with respect to this node.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createNode(
			eDomNodeType	eNodeType,
			uint				uiNameId,
			eNodeInsertLoc	eInsertLoc,
			DOMNode			nodeToReuse)
		{
			RCODE		rc;
			DOMNode	newNode;
			ulong		pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : 0;

			if ((rc = xflaim_DOMNode_createNode( m_pNode, m_db.getDb(),
				eNodeType, uiNameId, eInsertLoc, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			if (nodeToReuse == null)
			{
				newNode = new DOMNode( pNode, m_db);
			}
			else
			{
				newNode = nodeToReuse;
				newNode.setNodePtr( pNode, m_db);
			}
		
			return( newNode);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createNode(
			ulong				pNode,
			ulong				pDb,
			eDomNodeType	eNodeType,
			uint				uiNameId,
			eNodeInsertLoc	eInsertLoc,
			ref ulong		ppNode);

//-----------------------------------------------------------------------------
// createChildElement
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new element node and inserts it into the database in the
		/// as either the first or last child of the current node.  An existing
		/// <see cref="DOMNode"/> object can optionally be passed in, and it will be reused
		/// instead of a new object being allocated.
		/// </summary>
		/// <param name="uiChildElementNameId">
		/// The dictionary tag number that represents the node name. This name ID
		/// must exist in the dictionary before it can be used here.  The value
		/// may be one of the predefined ones, or it may be created by calling
		/// <see cref="Db.createElementDef"/>.
		/// </param>
		/// <param name="bFirstChild">
		/// Specifies whether the new element is to be created as a first or last child.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createChildElement(
			uint			uiChildElementNameId,
			bool			bFirstChild,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			DOMNode	newNode;
			ulong		pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : 0;

			if ((rc = xflaim_DOMNode_createChildElement( m_pNode, m_db.getDb(),
				uiChildElementNameId, (int)(bFirstChild ? 1 : 0), ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			if (nodeToReuse == null)
			{
				newNode = new DOMNode( pNode, m_db);
			}
			else
			{
				newNode = nodeToReuse;
				newNode.setNodePtr( pNode, m_db);
			}
		
			return( newNode);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createChildElement(
			ulong				pNode,
			ulong				pDb,
			uint				uiChildElementNameId,
			int				bFirstChild,
			ref ulong		ppNode);

//-----------------------------------------------------------------------------
// deleteNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Removes this node as well as all of it's descendants from the database.
		/// </summary>
		public void deleteNode()
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_deleteNode( m_pNode, m_db.getDb())) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_deleteNode(
			ulong				pNode,
			ulong				pDb);

//-----------------------------------------------------------------------------
// deleteChildren
//-----------------------------------------------------------------------------

		/// <summary>
		/// Removes the children of this node from the database.
		/// </summary>
		public void deleteChildren()
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_deleteChildren( m_pNode, m_db.getDb())) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_deleteChildren(
			ulong				pNode,
			ulong				pDb);

//-----------------------------------------------------------------------------
// getNodeType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Returns the type of node. 
		/// </summary>
		/// <returns>Type of node.</returns>
		public eDomNodeType getNodeType()
		{
			return( xflaim_DOMNode_getNodeType( m_pNode));
		}

		[DllImport("xflaim")]
		private static extern eDomNodeType xflaim_DOMNode_getNodeType(
			ulong				pNode);

//-----------------------------------------------------------------------------
// isDataLocalToNode
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if data for the current node is associated with the node, or
		/// with a child node.  Element nodes may not have data associated with them,
		/// but with child data nodes instead.
		/// </summary>
		/// <returns>
		/// Returns true if this node's data is associated with it, false otherwise.
		/// </returns>
		public bool isDataLocalToNode()
		{
			RCODE	rc;
			int	bLocal;

			if ((rc = xflaim_DOMNode_isDataLocalToNode( m_pNode, m_db.getDb(), out bLocal)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bLocal != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_isDataLocalToNode(
			ulong		pNode,
			ulong		pDb,
			out int	pbLocal);

//-----------------------------------------------------------------------------
// createAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Creates a new attribute node for this node.  Note that only element
		/// nodes are allowed to have attributes.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number that represents the attribute name. This name ID
		/// must exist in the dictionary before it can be used here.  The value
		/// may be one of the predefined ones, or it may be created by calling
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode createAttribute(
			uint			uiAttrNameId,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			DOMNode	newNode;
			ulong		pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : 0;

			if ((rc = xflaim_DOMNode_createAttribute( m_pNode, m_db.getDb(),
											uiAttrNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			if (nodeToReuse == null)
			{
				newNode = new DOMNode( pNode, m_db);
			}
			else
			{
				newNode = nodeToReuse;
				newNode.setNodePtr( pNode, m_db);
			}
		
			return( newNode);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_createAttribute(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			ref ulong	ppNode);

//-----------------------------------------------------------------------------
// getFirstAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the first attribute node associated with the current node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getFirstAttribute(
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			DOMNode	newNode;
			ulong		pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : 0;

			if ((rc = xflaim_DOMNode_getFirstAttribute( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			if (nodeToReuse == null)
			{
				newNode = new DOMNode( pNode, m_db);
			}
			else
			{
				newNode = nodeToReuse;
				newNode.setNodePtr( pNode, m_db);
			}
		
			return( newNode);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getFirstAttribute(
			ulong			pNode,
			ulong			pDb,
			ref ulong	ppNode);

//-----------------------------------------------------------------------------
// getLastAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the last attribute node associated with the current node.
		/// </summary>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getLastAttribute(
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			DOMNode	newNode;
			ulong		pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : 0;

			if ((rc = xflaim_DOMNode_getLastAttribute( m_pNode, m_db.getDb(),
				ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			if (nodeToReuse == null)
			{
				newNode = new DOMNode( pNode, m_db);
			}
			else
			{
				newNode = nodeToReuse;
				newNode.setNodePtr( pNode, m_db);
			}
		
			return( newNode);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLastAttribute(
			ulong			pNode,
			ulong			pDb,
			ref ulong	ppNode);

//-----------------------------------------------------------------------------
// getAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Retrieves the requested attribute node associated with this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number of the requested attribute.  The name id must
		/// exist in the dictionary before it can be used here.  The name id may be
		/// one of the predefined ones, or it may be created with
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		/// <param name="nodeToReuse">
		/// An existing <see cref="DOMNode"/> object can optionally be passed in, and
		/// it will be reused instead of a new object being allocated.
		/// </param>
		/// <returns>
		/// Returns a <see cref="DOMNode"/> object.
		/// </returns>
		public DOMNode getAttribute(
			uint			uiAttrNameId,
			DOMNode		nodeToReuse)
		{
			RCODE		rc;
			DOMNode	newNode;
			ulong		pNode = (nodeToReuse != null) ? nodeToReuse.getNode() : 0;

			if ((rc = xflaim_DOMNode_getAttribute( m_pNode, m_db.getDb(),
				uiAttrNameId, ref pNode)) != 0)
			{
				throw new XFlaimException( rc);
			}
			if (nodeToReuse == null)
			{
				newNode = new DOMNode( pNode, m_db);
			}
			else
			{
				newNode = nodeToReuse;
				newNode.setNodePtr( pNode, m_db);
			}
		
			return( newNode);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getAttribute(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			ref ulong	ppNode);

//-----------------------------------------------------------------------------
// deleteAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Deletes the specified attribute node associated with this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number of the attribute to delete.  The name id must
		/// exist in the dictionary before it can be used here.  The name id may be
		/// one of the predefined ones, or it may be created with
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		public void deleteAttribute(
			uint			uiAttrNameId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_deleteAttribute( m_pNode, m_db.getDb(),
				uiAttrNameId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_deleteAttribute(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId);

//-----------------------------------------------------------------------------
// hasAttribute
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the specified attribute exists for this node.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// The dictionary tag number of the attribute to check.  The name id must
		/// exist in the dictionary before it can be used here.  The name id may be
		/// one of the predefined ones, or it may be created with
		/// <see cref="Db.createAttributeDef"/>.
		/// </param>
		/// <returns>
		/// Returns true if the attribute exists, false otherwise.
		/// </returns>
		public bool hasAttribute(
			uint			uiAttrNameId)
		{
			RCODE		rc;
			int		bHasAttr;

			if ((rc = xflaim_DOMNode_hasAttribute( m_pNode, m_db.getDb(),
				uiAttrNameId, out bHasAttr)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasAttr != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasAttribute(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			out int		pbHasAttr);

//-----------------------------------------------------------------------------
// hasAttributes
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has any attributes.
		/// </summary>
		/// <returns>
		/// Returns true if the node has attributes, false otherwise.
		/// </returns>
		public bool hasAttributes()
		{
			RCODE		rc;
			int		bHasAttrs;

			if ((rc = xflaim_DOMNode_hasAttributes( m_pNode, m_db.getDb(),
				out bHasAttrs)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasAttrs != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasAttributes(
			ulong			pNode,
			ulong			pDb,
			out int		pbHasAttrs);

//-----------------------------------------------------------------------------
// hasNextSibling
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has a next sibling.
		/// </summary>
		/// <returns>
		/// Returns true if the node has a next sibling false otherwise.
		/// </returns>
		public bool hasNextSibling()
		{
			RCODE		rc;
			int		bHasNextSibling;

			if ((rc = xflaim_DOMNode_hasNextSibling( m_pNode, m_db.getDb(),
				out bHasNextSibling)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasNextSibling != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasNextSibling(
			ulong			pNode,
			ulong			pDb,
			out int		pbHasNextSibling);

//-----------------------------------------------------------------------------
// hasPreviousSibling
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has a previous sibling.
		/// </summary>
		/// <returns>
		/// Returns true if the node has a previous sibling false otherwise.
		/// </returns>
		public bool hasPreviousSibling()
		{
			RCODE		rc;
			int		bHasPreviousSibling;

			if ((rc = xflaim_DOMNode_hasPreviousSibling( m_pNode, m_db.getDb(),
				out bHasPreviousSibling)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasPreviousSibling != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasPreviousSibling(
			ulong			pNode,
			ulong			pDb,
			out int		pbHasPreviousSibling);

//-----------------------------------------------------------------------------
// hasChildren
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node has child nodes.
		/// </summary>
		/// <returns>
		/// Returns true if the node has child nodes, false otherwise.
		/// </returns>
		public bool hasChildren()
		{
			RCODE		rc;
			int		bHasChildren;

			if ((rc = xflaim_DOMNode_hasChildren( m_pNode, m_db.getDb(),
				out bHasChildren)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bHasChildren != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_hasChildren(
			ulong			pNode,
			ulong			pDb,
			out int		pbHasChildren);

//-----------------------------------------------------------------------------
// isNamespaceDecl
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determines if the node is a namespace declaration.
		/// </summary>
		/// <returns>
		/// Returns true if the node is a namespace declaration, false otherwise.
		/// </returns>
		public bool isNamespaceDecl()
		{
			RCODE		rc;
			int		bIsNamespaceDecl;

			if ((rc = xflaim_DOMNode_isNamespaceDecl( m_pNode, m_db.getDb(),
				out bIsNamespaceDecl)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( bIsNamespaceDecl != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_isNamespaceDecl(
			ulong			pNode,
			ulong			pDb,
			out int		pbIsNamespaceDecl);

//-----------------------------------------------------------------------------
// getParentId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the parent node ID for this node.
		/// </summary>
		/// <returns>
		/// Returns the parent node ID of this node.
		/// </returns>
		public ulong getParentId()
		{
			RCODE		rc;
			ulong		ulParentId;

			if ((rc = xflaim_DOMNode_getParentId( m_pNode, m_db.getDb(),
				out ulParentId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulParentId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getParentId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulParentId);

//-----------------------------------------------------------------------------
// getNodeId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of this node.
		/// </returns>
		public ulong getNodeId()
		{
			RCODE		rc;
			ulong		ulNodeId;

			if ((rc = xflaim_DOMNode_getNodeId( m_pNode, m_db.getDb(),
				out ulNodeId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulNodeId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNodeId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulNodeId);

//-----------------------------------------------------------------------------
// getDocumentId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the document ID for this node.
		/// </summary>
		/// <returns>
		/// Returns the document ID of this node.
		/// </returns>
		public ulong getDocumentId()
		{
			RCODE		rc;
			ulong		ulDocumentId;

			if ((rc = xflaim_DOMNode_getDocumentId( m_pNode, m_db.getDb(),
				out ulDocumentId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulDocumentId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getDocumentId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulDocumentId);

//-----------------------------------------------------------------------------
// getPrevSibId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the previous sibling for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the previous sibling for this node.
		/// </returns>
		public ulong getPrevSibId()
		{
			RCODE		rc;
			ulong		ulPrevSibId;

			if ((rc = xflaim_DOMNode_getPrevSibId( m_pNode, m_db.getDb(),
				out ulPrevSibId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulPrevSibId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getPrevSibId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulPrevSibId);

//-----------------------------------------------------------------------------
// getNextSibId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the next sibling for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the next sibling for this node.
		/// </returns>
		public ulong getNextSibId()
		{
			RCODE		rc;
			ulong		ulNextSibId;

			if ((rc = xflaim_DOMNode_getNextSibId( m_pNode, m_db.getDb(),
				out ulNextSibId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulNextSibId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNextSibId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulNextSibId);

//-----------------------------------------------------------------------------
// getFirstChildId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the first child for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the first child for this node.
		/// </returns>
		public ulong getFirstChildId()
		{
			RCODE		rc;
			ulong		ulFirstChildId;

			if ((rc = xflaim_DOMNode_getFirstChildId( m_pNode, m_db.getDb(),
				out ulFirstChildId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulFirstChildId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getFirstChildId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulFirstChildId);

//-----------------------------------------------------------------------------
// getLastChildId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the node ID of the last child for this node.
		/// </summary>
		/// <returns>
		/// Returns the node ID of the last child for this node.
		/// </returns>
		public ulong getLastChildId()
		{
			RCODE		rc;
			ulong		ulLastChildId;

			if ((rc = xflaim_DOMNode_getLastChildId( m_pNode, m_db.getDb(),
				out ulLastChildId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulLastChildId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getLastChildId(
			ulong			pNode,
			ulong			pDb,
			out ulong	pulLastChildId);

//-----------------------------------------------------------------------------
// getNameId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Gets the name ID of this node.
		/// </summary>
		/// <returns>
		/// Returns the name ID of this node.
		/// </returns>
		public uint getNameId()
		{
			RCODE		rc;
			uint		uiNameId;

			if ((rc = xflaim_DOMNode_getNameId( m_pNode, m_db.getDb(),
				out uiNameId)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiNameId);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_getNameId(
			ulong			pNode,
			ulong			pDb,
			out uint		puiNameId);

//-----------------------------------------------------------------------------
// setULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to an unsigned long integer.
		/// </summary>
		/// <param name="ulValue">
		/// Value to set into the node.
		/// </param>
		public void setULong(
			ulong		ulValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setULong( m_pNode, m_db.getDb(),
				ulValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to an unsigned long integer.
		/// </summary>
		/// <param name="ulValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setULong(
			ulong		ulValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setULong( m_pNode, m_db.getDb(),
				ulValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setULong(
			ulong			pNode,
			ulong			pDb,
			ulong			ulValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ulValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueULong(
			uint		uiAttrNameId,
			ulong		ulValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueULong( m_pNode, m_db.getDb(),
				uiAttrNameId, ulValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ulValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueULong(
			uint		uiAttrNameId,
			ulong		ulValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueULong( m_pNode, m_db.getDb(),
				uiAttrNameId, ulValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueULong(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			ulong			ulValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to a signed long integer.
		/// </summary>
		/// <param name="lValue">
		/// Value to set into the node.
		/// </param>
		public void setLong(
			long		lValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setLong( m_pNode, m_db.getDb(),
				lValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to a signed long integer.
		/// </summary>
		/// <param name="lValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setLong(
			long		lValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setLong( m_pNode, m_db.getDb(),
				lValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setLong(
			ulong			pNode,
			ulong			pDb,
			long			lValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="lValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueLong(
			uint		uiAttrNameId,
			long		lValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueLong( m_pNode, m_db.getDb(),
				uiAttrNameId, lValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed long integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="lValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueLong(
			uint		uiAttrNameId,
			long		lValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueLong( m_pNode, m_db.getDb(),
				uiAttrNameId, lValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueLong(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			long			lValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiValue">
		/// Value to set into the node.
		/// </param>
		public void setUInt(
			uint		uiValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setUInt( m_pNode, m_db.getDb(),
				uiValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setUInt(
			uint		uiValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setUInt( m_pNode, m_db.getDb(),
				uiValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setUInt(
			ulong			pNode,
			ulong			pDb,
			uint			uiValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="uiValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueUInt(
			uint		uiAttrNameId,
			uint		uiValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueUInt( m_pNode, m_db.getDb(),
				uiAttrNameId, uiValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to an unsigned integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="uiValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueUInt(
			uint		uiAttrNameId,
			uint		uiValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueUInt( m_pNode, m_db.getDb(),
				uiAttrNameId, uiValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueUInt(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			uint			uiValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value of the node to a signed integer.
		/// </summary>
		/// <param name="iValue">
		/// Value to set into the node.
		/// </param>
		public void setInt(
			int		iValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setInt( m_pNode, m_db.getDb(),
				iValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value of the node to a signed integer.
		/// </summary>
		/// <param name="iValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setInt(
			int		iValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setInt( m_pNode, m_db.getDb(),
				iValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setInt(
			ulong			pNode,
			ulong			pDb,
			int			iValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="iValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueInt(
			uint		uiAttrNameId,
			int		iValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueInt( m_pNode, m_db.getDb(),
				uiAttrNameId, iValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a signed integer.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="iValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueInt(
			uint		uiAttrNameId,
			int		iValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueInt( m_pNode, m_db.getDb(),
				uiAttrNameId, iValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueInt(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			int			iValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		public void setString(
			string	sValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, 1, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether sValue is the last text to be appended to this
		/// node.  If false, then another call to setString is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setString will overwrite the what is currently stored in this node.
		/// </param>
		public void setString(
			string	sValue,
			bool		bLast)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, (int)(bLast ? 1 : 0), 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setString(
			string	sValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, 1, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a string.
		/// </summary>
		/// <param name="sValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether sValue is the last text to be appended to this
		/// node.  If false, then another call to setString is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setString will overwrite the what is currently stored in this node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setString(
			string	sValue,
			bool		bLast,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setString( m_pNode, m_db.getDb(),
				sValue, (int)(bLast ? 1 : 0), uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setString(
			ulong			pNode,
			ulong			pDb,
			[MarshalAs(UnmanagedType.LPWStr), In]
			string		sValue,
			int			bLast,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a string
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="sValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueString(
			uint		uiAttrNameId,
			string	sValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueString( m_pNode, m_db.getDb(),
				uiAttrNameId, sValue, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a string.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="sValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueString(
			uint		uiAttrNameId,
			string	sValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueString( m_pNode, m_db.getDb(),
				uiAttrNameId, sValue, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueString(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			[MarshalAs(UnmanagedType.LPWStr), In]
			string		sValue,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		public void setBinary(
			byte []	ucValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, 1, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether ucValue is the last text to be appended to this
		/// node.  If false, then another call to setBinary is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setBinary will overwrite the what is currently stored in this node.
		/// </param>
		public void setBinary(
			 byte []	ucValue,
			bool		bLast)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, (int)(bLast ? 1 : 0), 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setBinary(
			byte []	ucValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, 1, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the node to a byte array of binary data.
		/// </summary>
		/// <param name="ucValue">
		/// Value to set into the node.
		/// </param>
		/// <param name="bLast">
		/// Specifies whether ucValue is the last text to be appended to this
		/// node.  If false, then another call to setBinary is expected, and
		/// the new text will be appended to the text currently stored in this
		/// node.  If true, then no more text is expected and  another call to
		/// setBinary will overwrite the what is currently stored in this node.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setBinary(
			byte []	ucValue,
			bool		bLast,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setBinary( m_pNode, m_db.getDb(),
				ucValue, (uint)ucValue.Length, (int)(bLast ? 1 : 0), uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setBinary(
			ulong			pNode,
			ulong			pDb,
			[MarshalAs(UnmanagedType.LPArray), In]
			byte []		ucValue,
			uint			uiLen,
			int			bLast,
			uint			uiEncId);

//-----------------------------------------------------------------------------
// setAttributeValueBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Sets the value for the specified attribute of the node to a
		/// byte array of binary data.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ucValue">
		/// Value to set into the attribute.
		/// </param>
		public void setAttributeValueBinary(
			uint		uiAttrNameId,
			byte []	ucValue)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueBinary( m_pNode, m_db.getDb(),
				uiAttrNameId, ucValue, (uint)ucValue.Length, 0)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		/// <summary>
		/// Sets the value for the specified attribute of the node to a
		/// byte array of binary data.
		/// </summary>
		/// <param name="uiAttrNameId">
		/// Name id of attribute whose value is to be set.
		/// </param>
		/// <param name="ucValue">
		/// Value to set into the attribute.
		/// </param>
		/// <param name="uiEncId">
		/// Encryption definition to use to encrypt the value.  An
		/// encryption id of zero means that the value should not be
		/// encrypted.
		/// </param>
		public void setAttributeValueBinary(
			uint		uiAttrNameId,
			byte []	ucValue,
			uint		uiEncId)
		{
			RCODE		rc;

			if ((rc = xflaim_DOMNode_setAttributeValueBinary( m_pNode, m_db.getDb(),
				uiAttrNameId, ucValue, (uint)ucValue.Length, uiEncId)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DOMNode_setAttributeValueBinary(
			ulong			pNode,
			ulong			pDb,
			uint			uiAttrNameId,
			[MarshalAs(UnmanagedType.LPWStr), In]
			byte []		ucValue,
			uint			uiLen,
			uint			uiEncId);

	}
}
