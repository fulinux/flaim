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
	/// DOM Node types
	/// </summary>
	public enum eDomNodeType : int
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
	/// </summary>
	public enum eNodeInsertLoc : int
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
}
