//------------------------------------------------------------------------------
// Desc:	Corrupt Info Structure
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
// $Id: CORRUPTINFO.java 3111 2006-01-19 13:10:50 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;
import xflaim.DOMNode;
import xflaim.DataVector;

/**
 * To change the template for this generated type comment go to
 * Window&gt;Preferences&gt;Java&gt;Code Generation&gt;Code and Comments
 */
public final class CORRUPTINFO
{
	int				iErrCode;				// Zero means no error is being reported
	int				iErrLocale;
	int				iErrLfNumber;
	int				iErrLfType;
	int				iErrBTreeLevel;
	int				iErrBlkAddress;
	int				iErrParentBlkAddress;
	int				iErrElmOffset;
	long				lErrNodeId;
	DataVector		ErrIxKey;
	DOMNode			ErrNode;
	DataVector[]	ErrNodeKeyList;

	public static class LOCALE_CODES
	{
		public static final int		LOCALE_NONE				= 0;
		public static final int		LOCALE_LFH_LIST		= 1;
		public static final int		LOCALE_AVAIL_LIST		= 2;
		public static final int		LOCALE_B_TREE			= 3;
		public static final int		LOCALE_INDEX			= 4;	
	}
}
