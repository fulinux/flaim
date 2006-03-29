//------------------------------------------------------------------------------
// Desc:	Data Type
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
// $Id: FlmDataType.java 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * Provides enums for all of the possible data types a DOM node can hold.
 * NOTE: The values in this class must match *exactly* with the equivalent
 * #defines in xflaim.h
 */
public final class FlmDataType
{
	public static final int FLM_NODATA_TYPE		= 0;
	public static final int FLM_TEXT_TYPE			= 1;
	public static final int FLM_NUMBER_TYPE		= 2;
	public static final int FLM_BINARY_TYPE		= 3;

	public static final int FLM_NUM_OF_TYPES		= 4;
	public static final int FLM_UNKNOWN_TYPE		= 0xF;
}
