//------------------------------------------------------------------------------
// Desc:	Check Flags
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
// $Id: DbCheckFlags.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * The members of this class are used by <code>DbSystem.dbCheck</code>.
 * The values of these members must match exactly with the equivalent
 * #defines in xflaim.h.
 */
public final class DbCheckFlags 
{
	public static final int	FO_ONLINE					= 0x0020; 
	public static final int	FO_DO_LOGICAL_CHECK		= 0x0100;
	public static final int	FO_SKIP_DOM_LINK_CHECK	= 0x0400;
}
