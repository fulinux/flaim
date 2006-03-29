//------------------------------------------------------------------------------
// Desc:	Key Retrieve Flags
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
// $Id: KeyRetrieveFlags.java 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * To change the template for this generated type comment go to
 * Window->Preferences->Java->Code Generation->Code and Comments
 */
public final class KeyRetrieveFlags
{
	public static final int FO_INCL 				= 0x0010;
	public static final int FO_EXCL 				= 0x0020;
	public static final int FO_EXACT 			= 0x0040;
	public static final int FO_KEY_EXACT 		= 0x0080;
	public static final int FO_FIRST 			= 0x0100;
	public static final int FO_LAST 				= 0x0200;
	public static final int FO_MATCH_IDS 		= 0x0400;
}
