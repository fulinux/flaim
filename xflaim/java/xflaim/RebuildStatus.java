//------------------------------------------------------------------------------
// Desc:	RebuildStatus Interface
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

interface RebuildStatus 
{
	public int reportRebuild(
		int				iDoingFlag,
		boolean			bStartFlag,
		long				l64FileSize,
		long				lBytesExamined,
		long				lTotNodes,
		long				lNodesRecov,
		long				lDiscardedDocs);
		
	public int reportRebuildErr(
		int			iErrCode,
		int			iErrLocale,
		int			iErrLfNumber,
		int			iErrLfType,
		int			iErrBTreeLevel,
		int			iErrBlkAddress,
		int			iErrParentBlkAddress,
		int			iErrElmOffset,
		long			lErrNodeId);
}	

