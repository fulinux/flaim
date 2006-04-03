//------------------------------------------------------------------------------
// Desc:	Backup Status
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: BackupStatus.java 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * This interface allows XFlaim's backup subsystem to periodicly pass
 * information about the status of a backup operation (bytes completed and
 * bytes remaining) while the operation is running.  The implementor may do
 * anything it wants with the information, such as using it to update a
 * progress bar or simply ignoring it.
 */
public interface BackupStatus
{
	/**
	 * Called by XFlaim's backup subsystem to pass information back
	 * to the user
	 * @param lBytesToDo The number of bytes that have not been backed up yet
	 * @param lBytesDone The number of bytes that have been backed up.
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * backup operation to abort an XFLaimException to be thrown.  
	 */
	public int backupStatus(
		long	lBytesToDo,
		long	lBytesDone);
}
