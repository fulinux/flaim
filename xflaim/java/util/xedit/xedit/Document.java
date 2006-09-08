//------------------------------------------------------------------------------
// Desc:	Documents
//
// Tabs:	3
//
//		Copyright (c) 2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: Document.java 3120 2006-01-19 13:41:12 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xedit;

/**
 * The Document class represents a simple structure to hold essential details
 * about each document that is opened in XEdit.  This class overrides the 
 * equality testing method used by Vectors to test if one object is
 * equivalent to another.  Instances of this class are stored in a vector
 * during program execution.  The Vector.indexOf method is used to
 * locate the index of the Document in the vector.
 */
public class Document
{
	public long		m_lDocId;
	public int		m_iCollection;
	public String	m_sName;

	/**
	 * 
	 */
	public Document(
		String		sName,
		long		lDocId,
		int			iCollection)
	{
		m_sName = sName;
		m_lDocId = lDocId;
		m_iCollection = iCollection;
	}
	
	public String toString()
	{
		return m_sName;
	}
	
	/**
	 * Overrides the Object.equals(Object) method to compare
	 * two Document objects for equivalence.  They need not be the 
	 * same object, they just have to be internally identical.
	 */
	public boolean equals(Object doc)
	{
		Document d = (Document)doc;
		
		if (m_lDocId == d.m_lDocId)
		{
			if (m_iCollection == d.m_iCollection)
			{
				if (m_sName.equals(d.m_sName))
				{
					return true;
				}
			}
		}
		return false;
	}
}
