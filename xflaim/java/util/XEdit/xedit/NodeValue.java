//------------------------------------------------------------------------------
// Desc:	Node Value
//
// Tabs:	3
//
//		Copyright (c) 2004-2006 Novell, Inc. All Rights Reserved.
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
// $Id: NodeValue.java 3120 2006-01-19 13:41:12 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xedit;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public class NodeValue
{
	String		m_sValue;
	/**
	 * 
	 */
	public NodeValue(
		String		sValue)
	{
		m_sValue = sValue;
	}
	
	public void setValue(
		String		sValue)
	{
		m_sValue = sValue;
	}
	
	public String getValue()
	{
		return m_sValue;
	}

}
