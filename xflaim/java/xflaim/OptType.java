//------------------------------------------------------------------------------
// Desc:	OptType
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

package xflaim;

/**
 * Provides list of optimization types for queries
 */

public final class OptType
{
	public static final int XFLM_QOPT_NONE							= 0;
	public static final int XFLM_QOPT_USING_INDEX				= 1;
	public static final int XFLM_QOPT_FULL_COLLECTION_SCAN	= 2;
	public static final int XFLM_QOPT_SINGLE_NODE_ID			= 3;
	public static final int XFLM_QOPT_NODE_ID_RANGE				= 4;
}

