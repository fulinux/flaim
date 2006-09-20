//------------------------------------------------------------------------------
// Desc:	Db Rebuild Status
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

using System;
using System.Runtime.InteropServices;

namespace xflaim
{

	/// <summary>
	/// Phases of a rebuild operation.
	/// </summary>
	public enum RebuildPhase
	{
		/// <summary>Determining block size</summary>
		REBUILD_GET_BLK_SIZ		= 1,
		/// <summary>Recovering the dictionary</summary>
		REBUILD_RECOVER_DICT		= 2,
		/// <summary>Recovering non-dictionary data</summary>
		REBUILD_RECOVER_DATA		= 3
	}

	/// <summary>
	/// Class that reports progress information for a <see cref="DbSystem.dbRebuild"/> operation.
	/// It is returned in the <see cref="DbRebuildStatus.reportRebuild"/> method.
	/// IMPORTANT NOTE: This structure needs to stay in sync with the XFLM_REBUILD_INFO
	/// structure defined in xflaim.h
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public class XFLM_REBUILD_INFO
	{
		/// <summary>Current phase of the rebuild operation</summary>
		public RebuildPhase	eDoingFlag;
		/// <summary>
		/// Indicates whether we are just starting this phase of the operation.  Value
		/// will be non-zero if just starting, zero otherwise.
		/// </summary>
		public int				bStartFlag;
		/// <summary>
		/// Size of the database in bytes.
		/// </summary>
		public ulong			ulDatabaseSize;
		/// <summary>
		/// Total bytes read so far.
		/// </summary>
		public ulong			ulBytesExamined;
		/// <summary>
		/// Total DOM nodes found so far.
		/// </summary>
		public ulong			ulTotNodes;
		/// <summary>
		/// Total DOM nodes recovered.
		/// </summary>
		public ulong			ulNodesRecov;
		/// <summary>
		/// Total DOM nodes discarded.
		/// </summary>
		public ulong			ulDiscardedDocs;
	}

	/// <summary>
	/// This interface allows XFlaim to periodically pass information back to the
	/// client about the status of an ongoing database rebuild operation.  The
	/// implementor may do anything it wants with the information, such as write
	/// it to a log file or display it on the screen.
	/// </summary>
	public interface DbRebuildStatus 
	{

		/// <summary>
		/// Called by <see cref="DbSystem.dbRebuild"/> to report progress of the
		/// rebuild operation.
		/// </summary>
		/// <param name="rebuildInfo">
		/// This object contains information about the progress of the
		/// rebuild operation.
		/// </param>
		/// <returns>
		/// If the implementation object returns anything but RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRebuild"/> operation will abort and throw an
		/// <see cref="XFlaimException"/>
		/// </returns>
		RCODE reportRebuild(
			XFLM_REBUILD_INFO	rebuildInfo);
		
		/// <summary>
		/// Called by <see cref="DbSystem.dbRebuild"/> to report corruptions found by the
		/// rebuild operation.
		/// </summary>
		/// <param name="corruptInfo">
		/// Information about the corruption is contained in this object.
		/// </param>
		/// <returns>
		/// If the implementation object returns anything but RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRebuild"/> operation will abort and throw an
		/// <see cref="XFlaimException"/>
		/// </returns>
		RCODE reportRebuildErr(
			XFLM_CORRUPT_INFO	corruptInfo);
	}
}
