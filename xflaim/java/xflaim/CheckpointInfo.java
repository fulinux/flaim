//------------------------------------------------------------------------------
// Desc:	CheckpointInfo Class
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
 * The IndexState class provides members that give an index's current state.
 */
public class CheckpointInfo 
{
	public boolean		bRunning;
	public int			iRunningTime;
	public boolean		bForcingCheckpoint;
	public int			iForceCheckpointRunningTime;
	public int			iForceCheckpointReason;
	public boolean		bWritingDataBlocks;
	public int			iLogBlocksWritten;
	public int			iDataBlocksWritten;
	public int			iDirtyCacheBytes;
	public int			iBlockSize;
	public int			iWaitTruncateTime;
	
	private static native void initIDs();
}

