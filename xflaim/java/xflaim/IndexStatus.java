//------------------------------------------------------------------------------
// Desc:	IndexStatus Class
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
public class IndexStatus 
{
	public int			iIndexNum;
	public int			iState;						// Should be one of the values defined in IndexState class
	public int			iStartTime;					// Start time of the offline process, or zero.
	public long			lLastDocumentIndexed;	// Value of the last document ID that was processed, unless -1.
															// If -1, the index is online.
	public long			lKeysProcessed;			// Keys processed in indexing thread.
	public long			lDocumentsProcessed;		// Documents processed in indexing thread.
	public long			lTransactions;				// Number of transactions started by indexing thread.
	
	private static native void initIDs();
}

