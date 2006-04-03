//------------------------------------------------------------------------------
// Desc:	Db Rename Status
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
// $Id: DbRenameStatus.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * This interface alows XFlaim to periodically pass information back to the
 * client about the status of an ongoing database rename operation.  The
 * implementor may do anything it wants with the information, such as write
 * it to a log file or display it on the screen.
 */
public interface DbRenameStatus
{
	
	/**
	 * Called after each file is renamed.
	 * @param sSrcFileName The old name of the file.
	 * @param sDstFileName The new name of the file.
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * rename operation to abort and an XFLaimException to be thrown. 
	 */
	public int dbRenameStatus(
		String	sSrcFileName,
		String	sDstFileName);
}
