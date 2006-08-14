//------------------------------------------------------------------------------
// Desc:	CacheInfo Class
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
 * The CacheInfo class provides members that give cache information
 */
public class CacheInfo 
{
	public int				iMaxBytes;
	public int				iTotalBytesAllocated;
	public boolean			bDynamicCacheAdjust;
	public int				iCacheAdjustPercent;
	public int				iCacheAdjustMin;
	public int				iCacheAdjustMax;
	public int				iCacheAdjustMinToLeave;
	public int				iDirtyCount;
	public int				iDirtyBytes;
	public int				iNewCount;
	public int				iNewBytes;
	public int				iLogCount;
	public int				iLogBytes;
	public int				iFreeCount;
	public int				iFreeBytes;
	public int				iReplaceableCount;
	public int				iReplaceableBytes;
	public boolean			bPreallocatedCache;
	public CacheUsage		BlockCache;
	public CacheUsage		NodeCache;
	
	private static native void initIDs();
}

